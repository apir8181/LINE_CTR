/**
 * Modify LINE for CTR click prediction.
 * Input file should follow the format:
 * - line n: node_id friend_num ad_num 
 * - line n+1: friend_1 friend_2 ... friend_m
 * - line n+2: ad_1 ad_1_label ad_2 ad_2_label ... ad_m'_label
 */

#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

#define NEG_SAMPLING_POWER .75
#define SIGMOID_BOUND 10
#define DISPLAY_ITER 10000

const char *data_file = "";
const char *X_output_file = "./X_embed.txt";
const char *Y_output_file = "./Y_embed.txt";
int D = 100;
int K = 5;
int num_threads = 1;
int num_epoch = 1;
float lambda = 0;
float lr = .025;

int num_nodes, num_ads;
float *X, *Y, *DX, *DY;

long* threads_offset;

std::uniform_int_distribution<int> alias_index_sampler;
std::uniform_real_distribution<float> alias_prob_sampler(0, 1);
int *alias_share;
float *node_degree;
float *alias_prob;

void ReadData() {
    FILE* pFILE = fopen(data_file, "r");
    if (pFILE == NULL) {
        printf("Error: file %s not found\n", data_file);
        exit(1);
    }

    num_nodes = 0;
    num_ads = 0;

    int node_id, friend_num, ad_num;
    while (fscanf(pFILE, "%d %d %d", &node_id, &friend_num, &ad_num) != EOF) {
        num_nodes = std::max(num_nodes, node_id + 1);

        int friend_id, ad_id, ad_label;
        for (int i = 0; i < friend_num; ++ i) {
            fscanf(pFILE, "%d", &friend_id);
            num_nodes = std::max(num_nodes, friend_id + 1);
        }
        
        for (int i = 0; i < ad_num; ++ i) {
            fscanf(pFILE, "%d %d", &ad_id, &ad_label);
            num_ads = std::max(num_ads, ad_id + 1); 
        }
    }

    printf("nodes num: %d, ads num: %d\n", num_nodes, num_ads);
    fclose(pFILE);
}

void InitThreadsOffset() {
    FILE* pFILE = fopen(data_file, "r");
    threads_offset = new long[num_threads];
    int nodes_per_thread = num_nodes / num_threads;
    for (int i = 0; i < num_nodes; ++ i) {
        if (i % nodes_per_thread == 0) threads_offset[i / nodes_per_thread] = ftell(pFILE);
        int node_id, friend_num, ad_num;
        int friend_id, ad_id, ad_label;
        fscanf(pFILE, "%d %d %d", &node_id, &friend_num, &ad_num);
        for (int i = 0; i < friend_num; ++ i) fscanf(pFILE, "%d", &friend_id);
        for (int i = 0; i < ad_num; ++ i) fscanf(pFILE, "%d %d", &ad_id, &ad_label);
    }
    fclose(pFILE);
}

void InitAliasTable() {
    assert(num_nodes != 0 && num_ads != 0);
    FILE* pFILE = fopen(data_file, "r");
    alias_share = new (std::nothrow) int[num_nodes];
    node_degree = new (std::nothrow) float[num_nodes];
    alias_prob = new (std::nothrow) float[num_nodes];
    assert(alias_share != NULL && node_degree != NULL && alias_prob != NULL);

    int node_id, friend_num, ad_num;
    while (fscanf(pFILE, "%d %d %d", &node_id, &friend_num, &ad_num) != EOF) {
        node_degree[node_id] = pow(friend_num, NEG_SAMPLING_POWER);
        int friend_id, ad_id, ad_label;
        for (int i = 0; i < friend_num; ++ i) fscanf(pFILE, "%d", &friend_id);
        for (int i = 0; i < ad_num; ++ i) fscanf(pFILE, "%d %d", &ad_id, &ad_label);
    }

    float sum = 0;
    std::vector<int> smaller, larger;
    for (int i = 0; i < num_nodes; ++ i) sum += node_degree[i]; 
    for (int i = 0; i < num_nodes; ++ i) {
        alias_prob[i] = node_degree[i] * num_nodes / sum; 
        alias_share[i] = i;
        if (alias_prob[i] < 1.0) smaller.push_back(i);
        else larger.push_back(i);
    }
    
    while (smaller.size() > 0 && larger.size() > 0) {
        int small = smaller.back(); smaller.pop_back();
        int large = larger.back(); larger.pop_back();
        alias_share[small] = large;
        alias_prob[large] -= 1.0 - alias_prob[small];
        if (alias_prob[large] < 1.0) smaller.push_back(large);
        else larger.push_back(small);
    }

    alias_index_sampler = std::uniform_int_distribution<int>(0, num_nodes - 1);
    fclose(pFILE);
}

int AliasSample(std::default_random_engine& gen) {
    int index = alias_index_sampler(gen);
    float prob = alias_prob_sampler(gen);
    if (prob < alias_prob[index]) return index;
    else return alias_share[index];
}

void InitEmbeddingMatrix() {
    X = new (std::nothrow) float[num_nodes * D];
    DX = new (std::nothrow) float[num_nodes * D];
    Y = new (std::nothrow) float[num_ads * D];
    DY = new (std::nothrow) float[num_ads * D];

    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(-.5f / D, .5f / D);

    for (int i = 0; i < num_nodes * D; ++ i) { X[i] = dist(gen); DX[i] = 0; }
    for (int i = 0; i < num_ads * D; ++ i) { Y[i] = dist(gen); DY[i] = 0; }
}

float SigmoidInnerProduct(float* x, float* y) {
    float ip = 0;
    for (int i = 0; i < D; ++ i) ip += x[i] * y[i];
    if (ip >= SIGMOID_BOUND) ip = SIGMOID_BOUND;
    if (ip <= -SIGMOID_BOUND) ip = -SIGMOID_BOUND;
    return (float)( 1.0 / (1.0 + exp(-ip)) );
}

void AccumulateGradient(float* dx, float *y, float weight) {
    for (int i = 0; i < D; ++ i) dx[i] += weight * y[i];
}

void Update(float *x, float *dx) {
    for (int i = 0; i < D; ++ i) {
        x[i] += lr * dx[i];
        dx[i] = 0;
    }
}

float TrainNbrPair(int node_id, int friend_id, std::default_random_engine& gen) {
    float loss = 0;
    std::vector<int> negatives(K);

    // postive friend
    float sigmoid = SigmoidInnerProduct(&X[node_id * D], &X[friend_id * D]);
    AccumulateGradient(&DX[node_id * D], &X[friend_id * D], 1 - sigmoid);
    AccumulateGradient(&DX[friend_id * D], &X[node_id * D], 1 - sigmoid);
    loss += -std::log(sigmoid);
    
    // negative friend
    for (int i = 0; i < K; ++ i) {
        negatives[i] = friend_id;
        while (negatives[i] == friend_id || negatives[i] == node_id) negatives[i] = AliasSample(gen);
        sigmoid = SigmoidInnerProduct(&X[node_id * D], &X[negatives[i] * D]);
        AccumulateGradient(&DX[node_id * D], &X[negatives[i] * D], -sigmoid);
        AccumulateGradient(&DX[negatives[i] * D], &X[node_id * D], -sigmoid);
        loss += -std::log(1 - sigmoid);
    }
 
    Update(&X[node_id * D], &DX[node_id * D]);
    Update(&X[friend_id * D], &DX[friend_id * D]);
    for (int i = 0; i < K; ++ i) Update(&X[negatives[i] * D], &DX[negatives[i] * D]);
    
    return loss;
}

float TrainAdPair(int node_id, int ad_id, int ad_label) {
    float sigmoid = SigmoidInnerProduct(&X[node_id * D], &Y[ad_id * D]);
    AccumulateGradient(&DX[node_id * D], &Y[ad_id * D], lambda * (ad_label - sigmoid));
    AccumulateGradient(&DY[ad_id * D], &X[node_id * D], lambda * (ad_label - sigmoid));
    float loss = ad_label == 0 ? -std::log(1 - sigmoid) : -std::log(sigmoid);
    Update(&X[node_id * D], &DX[node_id * D]);
    Update(&Y[ad_id * D], &DY[ad_id * D]);
    return loss;
}

void TrainNext(FILE* pFILE, std::default_random_engine& gen, 
        float& line_loss, int& line_count,
        float& ctr_loss, int& ctr_count) {
    int node_id, friend_num, ad_num;
    fscanf(pFILE, "%d %d %d", &node_id, &friend_num, &ad_num);

    std::vector<int> friends(friend_num), ads(ad_num), ads_label(ad_num);
    for (int i = 0; i < friend_num; ++ i) fscanf(pFILE, "%d", &friends[i]);
    for (int i = 0; i < ad_num; ++ i) fscanf(pFILE, "%d %d", &ads[i], &ads_label[i]);

    for (int i = 0; i < friend_num; ++ i) {
        line_loss += TrainNbrPair(node_id, friends[i], gen);
        line_count ++;
    }

    for (int i = 0; i < ad_num; ++ i) {
        ctr_loss += TrainAdPair(node_id, ads[i], ads_label[i]);
        ctr_count ++;
    }
}

void TrainThreadMain(int thread_idx) {
    float line_loss = 0, ctr_loss = 0;
    int line_count = 0, ctr_count = 0;
    int iter = 0;
    std::default_random_engine gen;

    for (int t = 0; t < num_epoch; ++ t) {
        FILE* pFILE = fopen(data_file, "r");
        fseek(pFILE, threads_offset[thread_idx], SEEK_SET);

        int nodes_thread = num_nodes / num_threads;
        if (thread_idx == num_threads - 1) nodes_thread = num_nodes - thread_idx * nodes_thread;

        for (int i = 0; i < nodes_thread; ++ i) {
            TrainNext(pFILE, gen, line_loss, line_count, ctr_loss, ctr_count);
            if (iter % DISPLAY_ITER == 0) {
                float a = line_loss / line_count;
                float b = ctr_count > 0 ? ctr_loss / ctr_count : 0;
                float c = a + lambda * b;
                float progress = 1.0f * iter / (nodes_thread * num_epoch);
                printf("[Thread-%d] iter %d, progress %f, total loss: %f, line loss: %f, ctr loss: %f\n",
                    thread_idx, iter, progress, c, a, b);
                line_loss = ctr_loss = 0;
                line_count = ctr_count = 0;
            }
            iter ++;
        }

        fclose(pFILE);
    }
}

void Output() {
    FILE* pFILE = fopen(X_output_file, "w");
    fprintf(pFILE, "%d %d\n", num_nodes, D);
    for (int i = 0; i < num_nodes; ++ i) {
        fprintf(pFILE, "%d", i);
        for (int j = 0; j < D; ++ j) fprintf(pFILE, " %f", X[i * D + j]);
        fprintf(pFILE, "\n");
    }
    fclose(pFILE);

    pFILE = fopen(Y_output_file, "w");
    fprintf(pFILE, "%d %d\n", num_ads, D);
    for (int i = 0; i < num_ads; ++ i) {
        fprintf(pFILE, "%d", i);
        for (int j = 0; j < D; ++ j) fprintf(pFILE, " %f", Y[i * D + j]);
        fprintf(pFILE, "\n");
    }
    fclose(pFILE);
}

void ParseArgs(int argc, char** argv) {
    for (int i = 1; i < argc; i ++) {
        if (!strcmp(argv[i], "-data_file")) data_file = argv[++i];
        else if (!strcmp(argv[i], "-X_output_file")) X_output_file = argv[++i];
        else if (!strcmp(argv[i], "-Y_output_file")) Y_output_file = argv[++i];
        else if (!strcmp(argv[i], "-embedding_size")) D = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-negative_size")) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-num_threads")) num_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-num_epoch")) num_epoch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-lambda")) lambda = atof(argv[++i]);
        else if (!strcmp(argv[i], "-init_learning_rate")) lr = atof(argv[++i]);
    }

    printf("Parameters:\n");
    printf("\tdata_file: %s\n", data_file);
    printf("\tX_output_file: %s\n", X_output_file);
    printf("\tY_output_file: %s\n", Y_output_file);
    printf("\tembedding_size: %d\n", D);
    printf("\tnegative_size: %d\n", K);
    printf("\nnum_threads: %d\n", num_threads);
    printf("\nnum_epoch: %d\n", num_epoch);
    printf("\nlambda: %f\n", lambda);
    printf("\ninit_learning_rate:%f\n", lr);
}

int main(int argc, char** argv) {
    ParseArgs(argc, argv);
    
    printf("----- Read Data\n");
    ReadData();

    printf("----- Init Threads Offset\n");
    InitThreadsOffset();

    printf("----- Init Alias Table\n");
    InitAliasTable();

    printf("----- Init Embedding Matrix\n");
    InitEmbeddingMatrix();

    printf("----- Start training\n");
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_threads; ++ i) TrainThreadMain(i);
    
    printf("----- Output embedding matrix\n");
    Output();

    return 0;
}
