//***********************************************************************************
// 2018.04.01 created by Zexlus1126
//
//    Example 002
// This is a simple demonstration on calculating merkle root from merkle branch 
// and solving a block (#286819) which the information is downloaded from Block Explorer 
//***********************************************************************************

#include <iostream>
#include <fstream>
#include <string>

#include <cstdio>
#include <cstring>

#include <cassert>

#include "sha256.h"

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 2048

////////////////////////   Block   /////////////////////

typedef struct __align__(8) _block {
    unsigned int version;
    unsigned char prevhash[32];
    unsigned char merkle_root[32];
    unsigned int ntime;
    unsigned int nbits;
    unsigned int nonce;
} HashBlock;

typedef struct __align__(8) _mining_result {
    int found;
    unsigned int nonce;
} MiningResult;

////////////////////////   Utils   ///////////////////////

// Add error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

//convert one hex-codec char to binary
unsigned char decode(unsigned char c)
{
    switch(c)
    {
        case 'a':
            return 0x0a;
        case 'b':
            return 0x0b;
        case 'c':
            return 0x0c;
        case 'd':
            return 0x0d;
        case 'e':
            return 0x0e;
        case 'f':
            return 0x0f;
        case '0' ... '9':
            return c-'0';
    }
}


// convert hex string to binary
//
// in: input string
// string_len: the length of the input string
//      '\0' is not included in string_len!!!
// out: output bytes array
void convert_string_to_little_endian_bytes(unsigned char* out, char *in, size_t string_len)
{
    assert(string_len % 2 == 0);

    size_t s = 0;
    size_t b = string_len/2-1;

    for(s, b; s < string_len; s+=2, --b)
    {
        out[b] = (unsigned char)(decode(in[s])<<4) + decode(in[s+1]);
    }
}

// print out binary array (from highest value) in the hex format
void print_hex(unsigned char* hex, size_t len)
{
    for(int i=0;i<len;++i)
    {
        printf("%02x", hex[i]);
    }
}


// print out binar array (from lowest value) in the hex format
void print_hex_inverse(unsigned char* hex, size_t len)
{
    for(int i=len-1;i>=0;--i)
    {
        printf("%02x", hex[i]);
    }
}

__device__ int cuda_little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len)
{
    // compared from lowest bit
    for(int i=byte_len-1;i>=0;--i)
    {
        if(a[i] < b[i])
            return -1;
        else if(a[i] > b[i])
            return 1;
    }
    return 0;
}

void getline(char *str, size_t len, FILE *fp)
{

    int i=0;
    while( i<len && (str[i] = fgetc(fp)) != EOF && str[i++] != '\n');
    str[len-1] = '\0';
}

////////////////////////   Hash   ///////////////////////

void double_sha256(SHA256 *sha256_ctx, unsigned char *bytes, size_t len)
{
    SHA256 tmp;
    sha256(&tmp, (BYTE*)bytes, len);
    sha256(sha256_ctx, (BYTE*)&tmp, sizeof(tmp));
}


////////////////////   Merkle Root   /////////////////////


// calculate merkle root from several merkle branches
// root: output hash will store here (little-endian)
// branch: merkle branch  (big-endian)
// count: total number of merkle branch
// Added CUDA kernel for mining
// Modified kernel to avoid memcpy and use constant memory
__global__ void find_nonce(const HashBlock block, 
                          const unsigned char* target_hex,
                          unsigned int start_nonce,
                          MiningResult* result)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (result->found) return;
    
    HashBlock my_block = block;
    my_block.nonce = start_nonce + tid;
    
    SHA256 sha256_ctx;
    SHA256 tmp;
    
    // Double SHA256 (same as original double_sha256)
    sha256(&sha256_ctx, (const BYTE*)&my_block, sizeof(HashBlock));
    tmp = sha256_ctx;
    sha256(&sha256_ctx, (const BYTE*)&tmp, sizeof(SHA256));
    
    // Use same comparison as original code
    if(cuda_little_endian_bit_comparison(sha256_ctx.b, target_hex, 32) < 0)
    {
        if(atomicCAS(&result->found, 0, 1) == 0) {
            result->nonce = my_block.nonce;
        }
    }
}


void calc_merkle_root(unsigned char *root, int count, char **branch)
{
    // [Merkle root calculation remains exactly the same as original]
    size_t total_count = count;
    unsigned char *raw_list = new unsigned char[(total_count+1)*32];
    unsigned char **list = new unsigned char*[total_count+1];

    for(int i=0;i<total_count; ++i)
    {
        list[i] = raw_list + i * 32;
        convert_string_to_little_endian_bytes(list[i], branch[i], 64);
    }
    list[total_count] = raw_list + total_count*32;

    while(total_count > 1)
    {
        int i, j;
        if(total_count % 2 == 1)
        {
            memcpy(list[total_count], list[total_count-1], 32);
        }
        for(i=0, j=0;i<total_count;i+=2, ++j)
        {
            SHA256 sha256_ctx;
            SHA256 tmp;
            sha256(&tmp, list[i], 64);
            sha256(&sha256_ctx, (BYTE*)&tmp, sizeof(tmp));
            memcpy(list[j], &sha256_ctx, 32);
        }
        total_count = j;
    }
    memcpy(root, list[0], 32);
    delete[] raw_list;
    delete[] list;
}

void solve(FILE *fin, FILE *fout)
{
    // Keep original input parsing
    char version[9];
    char prevhash[65];
    char ntime[9];
    char nbits[9];
    int tx;
    char *raw_merkle_branch;
    char **merkle_branch;

    getline(version, 9, fin);
    getline(prevhash, 65, fin);
    getline(ntime, 9, fin);
    getline(nbits, 9, fin);
    fscanf(fin, "%d\n", &tx);
    printf("start hashing\n");

    raw_merkle_branch = new char [tx * 65];
    merkle_branch = new char *[tx];
    for(int i=0;i<tx;++i)
    {
        merkle_branch[i] = raw_merkle_branch + i * 65;
        getline(merkle_branch[i], 65, fin);
        merkle_branch[i][64] = '\0';
    }

    unsigned char merkle_root[32];
    calc_merkle_root(merkle_root, tx, merkle_branch);

    printf("merkle root(little): ");
    print_hex(merkle_root, 32);
    printf("\n");

    printf("merkle root(big):    ");
    print_hex_inverse(merkle_root, 32);
    printf("\n");

    printf("Block info (big): \n");
    printf("  version:  %s\n", version);
    printf("  pervhash: %s\n", prevhash);
    printf("  merkleroot: "); print_hex_inverse(merkle_root, 32); printf("\n");
    printf("  nbits:    %s\n", nbits);
    printf("  ntime:    %s\n", ntime);
    printf("  nonce:    ???\n\n");

    HashBlock block;

    convert_string_to_little_endian_bytes((unsigned char *)&block.version, version, 8);
    convert_string_to_little_endian_bytes(block.prevhash, prevhash, 64);
    memcpy(block.merkle_root, merkle_root, 32);
    convert_string_to_little_endian_bytes((unsigned char *)&block.nbits, nbits, 8);
    convert_string_to_little_endian_bytes((unsigned char *)&block.ntime, ntime, 8);
    block.nonce = 0;
    
    // Calculate target
    unsigned int exp = block.nbits >> 24;
    unsigned int mant = block.nbits & 0xffffff;
    unsigned char target_hex[32] = {};
    
    unsigned int shift = 8 * (exp - 3);
    unsigned int sb = shift / 8;
    unsigned int rb = shift % 8;
    
    // little-endian target calculation (same as original)
    target_hex[sb    ] = (mant << rb);
    target_hex[sb + 1] = (mant >> (8-rb));
    target_hex[sb + 2] = (mant >> (16-rb));
    target_hex[sb + 3] = (mant >> (24-rb));
    
    printf("Target value (big): ");
    print_hex_inverse(target_hex, 32);
    printf("\n");

    // In solve(), add before CUDA initialization:
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Initialize CUDA
     HashBlock* d_block;
    unsigned char* d_target;
    MiningResult* d_result;
    cudaMalloc(&d_block, sizeof(HashBlock));
    cudaMalloc(&d_target, 32);
    cudaMalloc(&d_result, sizeof(MiningResult));

    cudaMemcpy(d_block, &block, sizeof(HashBlock), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_hex, 32, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(MiningResult));

    // Mining loop
    MiningResult h_result = {0, 0};
    unsigned int start_nonce = 0;
    SHA256 sha256_ctx;
    // Start timing before mining loop
    CUDA_CHECK(cudaEventRecord(start));

    while (!h_result.found && start_nonce <= 0xFFFFFFFF - THREADS_PER_BLOCK * NUM_BLOCKS) {
        find_nonce<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            block, d_target, start_nonce, d_result
        );
        
        cudaDeviceSynchronize();
        cudaMemcpy(&h_result, d_result, sizeof(MiningResult), cudaMemcpyDeviceToHost);
        
        if(h_result.found) {
            block.nonce = h_result.nonce;
            double_sha256(&sha256_ctx, (unsigned char*)&block, sizeof(block));
            printf("Found Solution!!\n");
            printf("hash #%10u (big): ", block.nonce);
            print_hex_inverse(sha256_ctx.b, 32);
            printf("\n\n");
            break;
        }
        
        start_nonce += THREADS_PER_BLOCK * NUM_BLOCKS;
        if(start_nonce % 1000000 == 0) {
            block.nonce = start_nonce;
            double_sha256(&sha256_ctx, (unsigned char*)&block, sizeof(block));
            printf("hash #%10u (big): ", start_nonce);
            print_hex_inverse(sha256_ctx.b, 32);
            printf("\n");
        }
    }

    // After mining loop ends (after the while loop), add:
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("\nExecution time with NUM_BLOCKS=%d: %.2f ms\n", NUM_BLOCKS, milliseconds);
    printf("Mining throughput: %.2f MH/s\n", 
        (start_nonce / 1000000.0f) / (milliseconds / 1000.0f));
    // Output result in same format as original
    for(int i=0; i<4; ++i) {
        fprintf(fout, "%02x", ((unsigned char*)&block.nonce)[i]);
    }
    fprintf(fout, "\n");

    // Cleanup
    cudaFree(d_block);
    cudaFree(d_target);
    cudaFree(d_result);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: cuda_miner <in> <out>\n");
        return 1;
    }
    
    FILE *fin = fopen(argv[1], "r");
    FILE *fout = fopen(argv[2], "w");
    
    if (!fin || !fout) {
        fprintf(stderr, "Error opening files\n");
        return 1;
    }

    int totalblock = 0;
    if (fscanf(fin, "%d\n", &totalblock) != 1) {
        fprintf(stderr, "Error reading total blocks\n");
        fclose(fin);
        fclose(fout);
        return 1;
    }
    
    fprintf(fout, "%d\n", totalblock);

    for(int i = 0; i < totalblock; i++) {
        solve(fin, fout);
    }

    fclose(fin);
    fclose(fout);
    return 0;
}

