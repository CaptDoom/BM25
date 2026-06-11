#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <algorithm>
#include "roaring.h"

// Helper to load raw data from an npy file (handling both 32-bit and 64-bit data sizes if required)
template<typename T>
std::vector<T> load_npy(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filepath << std::endl;
        return {};
    }
    
    char magic[6];
    file.read(magic, 6);
    if (std::strncmp(magic, "\x93NUMPY", 6) != 0) {
        std::cerr << "Invalid NPY file: " << filepath << std::endl;
        return {};
    }
    
    char major, minor;
    file.read(&major, 1);
    file.read(&minor, 1);
    
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);
    
    std::string header(header_len, '\0');
    file.read(&header[0], header_len);
    
    size_t header_offset = 10 + header_len;
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    size_t data_size = file_size - header_offset;
    file.seekg(header_offset, std::ios::beg);
    
    // Check if the data type is 8-byte integer (int64) or 4-byte float/int32
    size_t element_size = sizeof(T);
    bool is_int64 = false;
    if (header.find("<i8") != std::string::npos || header.find("|i8") != std::string::npos) {
        element_size = 8;
        is_int64 = true;
    }
    
    size_t num_elements = data_size / element_size;
    if (is_int64) {
        std::vector<int64_t> temp_data(num_elements);
        file.read(reinterpret_cast<char*>(temp_data.data()), data_size);
        std::vector<T> data(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = static_cast<T>(temp_data[i]);
        }
        return data;
    } else {
        std::vector<T> data(num_elements);
        file.read(reinterpret_cast<char*>(data.data()), data_size);
        return data;
    }
}

#pragma pack(push, 1)
struct DocMetadata {
    uint32_t length;
    uint16_t flags;
    uint8_t dataset_id;
};
#pragma pack(pop)

struct BM25ScoringEngine {
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    std::vector<float> data;
    std::vector<DocMetadata> metadata;
    std::unordered_map<std::string, roaring_bitmap_t*> bitmaps;
    int num_docs;
};

bool load_bitmaps(BM25ScoringEngine* engine, const std::string& bin_path, const std::string& txt_path) {
    std::ifstream txt_file(txt_path);
    if (!txt_file.is_open()) {
        std::cerr << "Error opening bitmaps text file: " << txt_path << std::endl;
        return false;
    }
    
    std::vector<std::string> names;
    std::string name;
    while (std::getline(txt_file, name)) {
        // Strip carriage returns on Windows
        if (!name.empty() && name.back() == '\r') {
            name.pop_back();
        }
        if (!name.empty()) {
            names.push_back(name);
        }
    }
    txt_file.close();
    
    std::ifstream bin_file(bin_path, std::ios::binary);
    if (!bin_file.is_open()) {
        std::cerr << "Error opening bitmaps binary file: " << bin_path << std::endl;
        return false;
    }
    
    uint32_t num_bitmaps = 0;
    bin_file.read(reinterpret_cast<char*>(&num_bitmaps), 4);
    if (num_bitmaps != names.size()) {
        std::cerr << "Mismatch in bitmap count: names=" << names.size() << ", binary=" << num_bitmaps << std::endl;
        return false;
    }
    
    std::vector<uint64_t> offsets(num_bitmaps);
    bin_file.read(reinterpret_cast<char*>(offsets.data()), num_bitmaps * 8);
    
    for (uint32_t i = 0; i < num_bitmaps; ++i) {
        bin_file.seekg(offsets[i], std::ios::beg);
        
        size_t size = 0;
        if (i + 1 < num_bitmaps) {
            size = offsets[i+1] - offsets[i];
        } else {
            bin_file.seekg(0, std::ios::end);
            size_t file_len = bin_file.tellg();
            size = file_len - offsets[i];
        }
        
        std::vector<char> buffer(size);
        bin_file.seekg(offsets[i], std::ios::beg);
        bin_file.read(buffer.data(), size);
        
        roaring_bitmap_t* bm = roaring_bitmap_portable_deserialize_safe(buffer.data(), size);
        if (bm == nullptr) {
            std::cerr << "Failed to deserialize bitmap: " << names[i] << std::endl;
            return false;
        }
        engine->bitmaps[names[i]] = bm;
    }
    bin_file.close();
    return true;
}

extern "C" {
    __declspec(dllexport) void* create_engine(
        const char* indices_path,
        const char* indptr_path,
        const char* data_path,
        const char* metadata_bin_path,
        const char* bitmaps_bin_path,
        const char* bitmaps_txt_path,
        int num_docs
    ) {
        BM25ScoringEngine* engine = new BM25ScoringEngine();
        engine->indptr = load_npy<int32_t>(indptr_path);
        engine->indices = load_npy<int32_t>(indices_path);
        engine->data = load_npy<float>(data_path);
        engine->num_docs = num_docs;
        
        if (metadata_bin_path && std::strlen(metadata_bin_path) > 0) {
            std::ifstream file(metadata_bin_path, std::ios::binary);
            if (file.is_open()) {
                engine->metadata.resize(num_docs);
                file.read(reinterpret_cast<char*>(engine->metadata.data()), num_docs * sizeof(DocMetadata));
                file.close();
            }
        }
        
        if (bitmaps_bin_path && bitmaps_txt_path && std::strlen(bitmaps_bin_path) > 0 && std::strlen(bitmaps_txt_path) > 0) {
            load_bitmaps(engine, bitmaps_bin_path, bitmaps_txt_path);
        }
        
        return reinterpret_cast<void*>(engine);
    }

    __declspec(dllexport) void free_engine(void* engine_ptr) {
        if (engine_ptr) {
            BM25ScoringEngine* engine = reinterpret_cast<BM25ScoringEngine*>(engine_ptr);
            for (auto& pair : engine->bitmaps) {
                roaring_bitmap_free(pair.second);
            }
            delete engine;
        }
    }

    __declspec(dllexport) int search(
        void* engine_ptr,
        const int* query_term_ids,
        int num_terms,
        const char** filter_names,
        int num_filters,
        int top_k,
        int* out_doc_ids,
        float* out_scores
    ) {
        if (!engine_ptr) return 0;
        BM25ScoringEngine* engine = reinterpret_cast<BM25ScoringEngine*>(engine_ptr);
        
        roaring_bitmap_t* allowed_set = nullptr;
        bool has_filters = (num_filters > 0 && filter_names != nullptr);
        
        if (has_filters) {
            for (int i = 0; i < num_filters; ++i) {
                std::string filter_name = filter_names[i];
                auto it = engine->bitmaps.find(filter_name);
                if (it != engine->bitmaps.end()) {
                    if (allowed_set == nullptr) {
                        allowed_set = roaring_bitmap_copy(it->second);
                    } else {
                        roaring_bitmap_and_inplace(allowed_set, it->second);
                    }
                } else {
                    if (allowed_set != nullptr) roaring_bitmap_free(allowed_set);
                    return 0;
                }
            }
            if (allowed_set == nullptr) {
                return 0;
            }
        }
        
        if (has_filters && roaring_bitmap_is_empty(allowed_set)) {
            roaring_bitmap_free(allowed_set);
            return 0;
        }
        
        std::vector<uint32_t> allowed_docs;
        uint64_t allowed_size = 0;
        if (has_filters) {
            allowed_size = roaring_bitmap_get_cardinality(allowed_set);
            allowed_docs.resize(allowed_size);
            roaring_bitmap_to_uint32_array(allowed_set, allowed_docs.data());
        }
        
        // Dirty list score accumulation to avoid zeroing out array of size num_docs
        thread_local static std::vector<float> doc_scores;
        thread_local static std::vector<uint32_t> dirty_docs;
        
        if (doc_scores.size() < (size_t)engine->num_docs) {
            doc_scores.assign(engine->num_docs, 0.0f);
        }
        
        dirty_docs.clear();
        
        for (int i = 0; i < num_terms; ++i) {
            int term_id = query_term_ids[i];
            if (term_id < 0 || term_id >= (int)engine->indptr.size() - 1) {
                continue;
            }
            
            int32_t start = engine->indptr[term_id];
            int32_t end = engine->indptr[term_id + 1];
            int32_t n = end - start;
            if (n <= 0) continue;
            
            if (!has_filters) {
                for (int32_t idx = start; idx < end; ++idx) {
                    uint32_t doc_id = engine->indices[idx];
                    float score = engine->data[idx];
                    if (doc_scores[doc_id] == 0.0f) {
                        dirty_docs.push_back(doc_id);
                    }
                    doc_scores[doc_id] += score;
                }
            } else {
                // Card-adaptive set intersection
                if (n < (int32_t)allowed_size / 32) {
                    // Posting list is much smaller: iterate posting list and do lookup
                    for (int32_t idx = start; idx < end; ++idx) {
                        uint32_t doc_id = engine->indices[idx];
                        if (roaring_bitmap_contains(allowed_set, doc_id)) {
                            float score = engine->data[idx];
                            if (doc_scores[doc_id] == 0.0f) {
                                dirty_docs.push_back(doc_id);
                            }
                            doc_scores[doc_id] += score;
                        }
                    }
                } else {
                    // Both are large or allowed set is smaller: do two-pointer intersection
                    int32_t a_idx = 0;
                    int32_t p_idx = start;
                    while (a_idx < (int32_t)allowed_size && p_idx < end) {
                        uint32_t a_doc = allowed_docs[a_idx];
                        uint32_t p_doc = engine->indices[p_idx];
                        if (a_doc == p_doc) {
                            float score = engine->data[p_idx];
                            if (doc_scores[a_doc] == 0.0f) {
                                dirty_docs.push_back(a_doc);
                            }
                            doc_scores[a_doc] += score;
                            a_idx++;
                            p_idx++;
                        } else if (a_doc < p_doc) {
                            a_idx++;
                        } else {
                            p_idx++;
                        }
                    }
                }
            }
        }
        
        int num_scored = dirty_docs.size();
        if (num_scored == 0) {
            if (allowed_set) roaring_bitmap_free(allowed_set);
            return 0;
        }
        
        int return_count = std::min(top_k, num_scored);
        
        std::partial_sort(
            dirty_docs.begin(),
            dirty_docs.begin() + return_count,
            dirty_docs.end(),
            [](uint32_t doc1, uint32_t doc2) {
                float s1 = doc_scores[doc1];
                float s2 = doc_scores[doc2];
                if (s1 != s2) return s1 > s2;
                return doc1 < doc2;
            }
        );
        
        for (int i = 0; i < return_count; ++i) {
            uint32_t doc_id = dirty_docs[i];
            out_doc_ids[i] = doc_id;
            out_scores[i] = doc_scores[doc_id];
        }
        
        for (uint32_t doc_id : dirty_docs) {
            doc_scores[doc_id] = 0.0f;
        }
        
        if (allowed_set) {
            roaring_bitmap_free(allowed_set);
        }
        
        return return_count;
    }
}
