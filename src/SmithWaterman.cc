// #pragma GCC optimize("inline-small-functions")
#pragma GCC optimize("-ftree-vectorize")
#pragma GCC target("avx2")
#pragma GCC optimize(2)
#pragma GCC optimize(3,"Ofast","inline")
#include "SmithWaterman.hpp"
#include <omp.h>
#include <immintrin.h>
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <thread>
#include <stdexcept>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
SmithWaterman::SmithWaterman(const std::string& query_seq_path,
                             const std::string& target_seq_path) {
  read_seq(query_seq_path, query_seqs);
  read_seq(target_seq_path, target_seqs);
  
  query_seqs_size = query_seqs.size();
  assert(query_seqs_size >= 1);
  target_seqs_size = target_seqs.size();
  assert(target_seqs_size >= 1);
}

class ThreadPool {
public:
    ThreadPool(size_t num_threads);
    ~ThreadPool();

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type>;

private:
    std::vector<std::thread> workers;                 
    std::queue<std::function<void()>> tasks;          
    std::mutex queue_mutex;                           
    std::condition_variable condition;                
    bool stop;                                        

    void worker();                                    
};

ThreadPool::ThreadPool(size_t num_threads) : stop(false) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back([this]() { worker(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

void ThreadPool::worker() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            condition.wait(lock, [this]() { return stop || !tasks.empty(); });
            if (stop && tasks.empty()) return;
            task = std::move(tasks.front());
            tasks.pop();
        }
        task();
    }
}
std::vector<size_t> SmithWaterman::solve() {
  ThreadPool pool(4);
  // Iterate through the query sequences
  max_scores.resize(query_seqs.size()*target_seqs.size());
  size_t idx = 0;
  for (auto& query_seq : query_seqs) {
    size_t query_seq_length = query_seq.sequence.size();

    // Iterate throuth the target sequences
    for (auto& target_seq : target_seqs) {
      size_t target_seq_length = target_seq.sequence.size();
      // Pairwise-Alignment
      pool.enqueue([this, &query_seq, &target_seq, query_seq_length, target_seq_length, idx]() {
          this->pair_align(query_seq, target_seq, query_seq_length, target_seq_length, idx);
      });
      // pair_align(query_seq, target_seq, query_seq_length, target_seq_length, idx);
      ++idx;
      
    }
  }

  return max_scores;
}

void SmithWaterman::report() const {
  for (size_t i = 0; i < query_seqs_size; i++) {
    auto max_score_ptr =
        std::max_element(max_scores.cbegin() + i * target_seqs_size,
                         max_scores.cbegin() + (i + 1) * target_seqs_size);
    size_t max_score_idx = std::distance(max_scores.cbegin(), max_score_ptr);
    std::cout << "Current query sequence: " << query_seqs.at(i).description
              << std::endl;
    std::cout << "The most similar sequence: "
              << target_seqs.at(max_score_idx % target_seqs_size).description
              << std::endl;
    std::cout << "The Simiarity Score: " << *max_score_ptr << std::endl
              << std::endl;
  }
}

__m256i mm256_max_epi64(__m256i a, __m256i b) {
    __m256i mask = _mm256_cmpgt_epi64(a, b);
    return _mm256_blendv_epi8(b, a, mask);
}

void SmithWaterman::pair_align(FastaSequence& query_seq,
                               FastaSequence& target_seq,
                               size_t query_seq_length,
                               size_t target_seq_length,
                               size_t idx) {
  query_seq.sequence.resize(query_seq_length + 8);
  int32_t qLens=(query_seq_length - 1) / 8 + 1;
  __m256i vG = _mm256_set1_epi32(gap_score);
  __m256i vH,vMAX,_0;
  vH = vMAX = _0 = _mm256_set1_epi32(0);
  int32_t e[8];
  // std::vector<__m256i> vLoadH(qLens + 1,_mm256_set1_epi32(0)),vStoreH(qLens + 1,_mm256_set1_epi32(0));
  __m256i* vLoadH = (__m256i*)_mm_malloc((qLens + 1) * sizeof(__m256i), 32);
  __m256i* vStoreH = (__m256i*)_mm_malloc((qLens + 1) * sizeof(__m256i), 32);
  std::fill(vLoadH, vLoadH + qLens + 1, _0);
  std::fill(vStoreH, vStoreH + qLens + 1, _0);
  for(int32_t j(1);j <= target_seq_length; j++){
    __m256i vF = _mm256_set1_epi32(0);
    // __m256i shifted = _mm256_srli_si256(vStoreH[j], 4);

    // 将 vStoreH 的最左边位置填充为 0
    vH = _mm256_permutevar8x32_epi32(vStoreH[qLens], _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 0));
    vH = _mm256_insert_epi32(vH, 0, 0);
    std::swap(vLoadH,vStoreH);
    #pragma omp parallel for
    for(int32_t i(1);i <= qLens; i++){
      
      for(int32_t k(0); k < 8; k++){
        e[k] = (query_seq.sequence[i - 1 + k * qLens] == target_seq.sequence[j - 1]
                ? match_score
                : mismatch_score);
      }
      __m256i vM = _mm256_set_epi32(e[7], e[6], e[5], e[4], e[3], e[2], e[1], e[0]);
      vH = _mm256_add_epi32(vH,vM);
      vH = _mm256_max_epi32(vH,_mm256_add_epi32(vLoadH[i],vG));
      vH = _mm256_max_epi32(vH,vF);
      vH = _mm256_max_epi32(vH,_0);
      vMAX = _mm256_max_epi32(vMAX,vH);
      vStoreH[i] = vH;
      vF = _mm256_add_epi32(vH,vG);
      vH = vLoadH[i];
    }
    vF = _mm256_permutevar8x32_epi32(vF, _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 0));
    vF = _mm256_insert_epi32(vF, 0, 0);
    int32_t i(1);
    while(1){
      __m256i CmpRes = _mm256_cmpgt_epi32(vF, vStoreH[i]);
      int32_t mask = _mm256_movemask_epi8(CmpRes);
      if(!mask)break;
      vStoreH[i] = _mm256_max_epi32(vStoreH[i],vF);
      vF = _mm256_add_epi32(vF,vG);
      if(++i > qLens){
        vF = _mm256_permutevar8x32_epi32(vF, _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 0));
        vF = _mm256_insert_epi32(vF, 0, 0);
        i = 1;
      }
    }
  }
  int32_t values[8];
  _mm256_storeu_si256((__m256i*)values, vMAX);
  int32_t Max = 0;
  for(int i = 0; i < 8; i++)Max = std::max(Max,values[i]);
  // max_scores.emplace_back(Max);
  max_scores[idx] = Max;
  if(vLoadH != nullptr){
    _mm_free(vLoadH);
    vLoadH = nullptr;
  }
  if(vStoreH != nullptr){
    _mm_free(vStoreH);
    vStoreH = nullptr;
  }
}

int SmithWaterman::validate(const std::string& ref_path) {
  read_ref(ref_path, refs);
  if (refs == max_scores) {
    std::cout << "Result correct!!!" << std::endl;
    report();
    return 0;
  } else {
    std::cout << "Result not match!!!" << std::endl;
    std::cout << "Reference Scores: ";
    std::copy(refs.cbegin(), refs.cend(),
              std::ostream_iterator<size_t>(std::cout, " "));
    std::cout << std::endl;

    std::cout << "Calculated Scores: ";
    std::copy(max_scores.cbegin(), max_scores.cend(),
              std::ostream_iterator<size_t>(std::cout, " "));
    std::cout << std::endl;

    return -1;
  }
}

void SmithWaterman::read_seq(const std::string& seq_path,
                             std::vector<FastaSequence>& seqs) {
  // Open file
  std::ifstream seq_file{seq_path};
  if (!seq_file.is_open()) {
    std::cerr << "Error opening file: " << seq_path << std::endl;
    throw std::runtime_error("Failed to open sequence file");
  }

  std::string line;
  FastaSequence curr_seq;

  // Read seqs
  while (std::getline(seq_file, line)) {
    if (line[0] == '>') {
      if (!curr_seq.sequence.empty()) {
        seqs.emplace_back(curr_seq);
        curr_seq = FastaSequence();
      }
      curr_seq.description = line.substr(1);
    } else {
      curr_seq.sequence += line;
    }
  }
  if (!curr_seq.sequence.empty()) {
    seqs.emplace_back(curr_seq);
  }

  // Close file
  seq_file.close();
}

void SmithWaterman::read_ref(const std::string& ref_path,
                             std::vector<size_t>& refs) {
  std::ifstream ref_file{ref_path};
  if (!ref_file.is_open()) {
    std::cerr << "Error opening the reference file: " << ref_path << std::endl;
    throw std::runtime_error("Failed to open reference file");
  }

  std::string line;
  while (std::getline(ref_file, line)) {
    std::istringstream iss(line);
    size_t score;
    while (iss >> score) {
      refs.emplace_back(score);
    }
  }
  ref_file.close();
}
