__kernel void slowSimpleGemm(__global float *in1, __global float *in2, __global float *out,
                         unsigned int col1, unsigned int row1, unsigned int col2, unsigned int row2) {
    unsigned int row = get_global_id(0);
    unsigned int col = get_global_id(1);

    if (row < row1 && col < col2) {
        float acc = 0.0f;
        for (size_t i = 0; i < col1; i++) {
            acc += in1[row * col1 + i] * in2[i * col2 + col];
        }
        out[row * col2 + col] = acc;
    }
}

__kernel void simpleGemm(__global float *in1, __global float *in2, __global float *out,
                         unsigned int col1, unsigned int row1, unsigned int col2, unsigned int row2) {
    unsigned int row = get_global_id(1);
    unsigned int col = get_global_id(0);

    if (row < row1 && col < col2) {
        float acc = 0.0f;
        for (size_t i = 0; i < col1; i++) {
            acc += in1[row * col1 + i] * in2[i * col2 + col];
        }
        out[row * col2 + col] = acc;
    }
}

__kernel void slowOptGemm(__global float *in1, __global float *in2, __global float *out,
                      unsigned int col1, unsigned int row1, unsigned int col2, unsigned int row2) {
    #define BLOCK_SIZE 16
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    __local float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __local float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    float acc = 0.0f;

    const int numTiles = col1 / BLOCK_SIZE;
    for (int t = 0; t < numTiles; t++) {
        const int tiledRow = BLOCK_SIZE*t + row;
        const int tiledCol = BLOCK_SIZE*t + col;
        Asub[row][col] = in1[globalRow * col1 + tiledCol];
        Bsub[row][col] = in2[tiledRow*col2 + globalCol];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < BLOCK_SIZE; k++) {
            acc += Asub[row][k] * Bsub[k][col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    out[globalRow * col2 + globalCol] = acc;
}

__kernel void optGemm(__global float *in1, __global float *in2, __global float *out,
                      unsigned int col1, unsigned int row1, unsigned int col2, unsigned int row2) {
    #define BLOCK_SIZE 16
    const int row = get_local_id(1);
    const int col = get_local_id(0);
    const int globalRow = get_global_id(1);
    const int globalCol = get_global_id(0);

    __local float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __local float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    float acc = 0.0f;

    const int numTiles = col1 / BLOCK_SIZE;
    for (int t = 0; t < numTiles; t++) {
        const int tiledRow = BLOCK_SIZE*t + row;
        const int tiledCol = BLOCK_SIZE*t + col;
        Asub[row][col] = in1[globalRow * col1 + tiledCol];
        Bsub[row][col] = in2[tiledRow*col2 + globalCol];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < BLOCK_SIZE; k++) {
            acc += Asub[row][k] * Bsub[k][col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    out[globalRow * col2 + globalCol] = acc;
}
 
__kernel void slowImageGemm(__read_only image2d_t in1, __read_only image2d_t in2, __global float *out,
                            unsigned int col1, unsigned int row1, unsigned int col2, unsigned int row2) {
    unsigned int row = get_global_id(0);
    unsigned int col = get_global_id(1);
    
    if (row < row1 && col < col2) {
        float acc = 0.0f;

        for (int i = 0; i < col1; i++) {
            int2 coordIn1 = (int2)(i, row); 
            int2 coordIn2 = (int2)(col, i);
            acc += read_imagef(in1, coordIn1).x * read_imagef(in2, coordIn2).x;
        }
        out[row * col2 + col] = acc;
    }
}

__kernel void imageGemm(__read_only image2d_t in1, __read_only image2d_t in2, __global float *out,
                        unsigned int col1, unsigned int row1, unsigned int col2, unsigned int row2) {
    #define BLOCK_SIZE 16
    const int row = get_local_id(1);
    const int col = get_local_id(0);
    const int globalRow = get_global_id(1);
    const int globalCol = get_global_id(0);

    __local float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __local float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    float acc = 0.0f;

    const int numTiles = col1 / BLOCK_SIZE;
    for (int t = 0; t < numTiles; t++) {
        const int tiledRow = BLOCK_SIZE*t + row;
        const int tiledCol = BLOCK_SIZE*t + col;
        int2 coordIn1 = (int2)(tiledCol, globalRow); 
        int2 coordIn2 = (int2)(globalCol, tiledRow);
        Asub[row][col] = read_imagef(in1, coordIn1).x;
        Bsub[row][col] = read_imagef(in2, coordIn2).x;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < BLOCK_SIZE; k++) {
            acc += Asub[row][k] * Bsub[k][col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    out[globalRow * col2 + globalCol] = acc;
}