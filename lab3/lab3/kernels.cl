__kernel void simpleGemm(__global float *in1, __global float *in2, __global float *out,
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

__kernel void optGemm(__global float *in1, __global float *in2, __global float *out,
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

__constant sampler_t Sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_LINEAR;
__kernel void imageGemm(__read_only image2d_t in1, __read_only image2d_t in2, __write_only image2d_t out,
                         unsigned int col1, unsigned int row1, unsigned int col2, unsigned int row2) {
    unsigned int row = get_global_id(0);
    unsigned int col = get_global_id(1);

    if (row == 0 && col == 0) {
        for (int row = 0; row < row1; row++) {
            for (int col = 0; col < col1; col++) {
                int2 coordIn1 = (int2)(row, col);
                float4 currentElIn1 = read_imagef(in1, Sampler, coordIn1);
                printf("%d %d %d %d | ", currentElIn1.x, currentElIn1.y, currentElIn1.z, currentElIn1.w);
            }
            printf("\n");
        }
    }
    /*int2 coordinates = (int2)(row, col); 
    printf("COORD: %d %d \n", coordinates.x, coordinates.y);
    int2 coordIn1 = (int2)(0, 0);
    int2 coordIn2 = (int2)(0, 0);
    float4 currentElIn1 = read_imagef(in1, Sampler, coordIn1);
    float4 currentElIn2 = read_imagef(in2, Sampler, coordIn2);
    printf("IN1: %d %d %d %d\n", currentElIn1.x, currentElIn1.y, currentElIn1.z, currentElIn1.w);
    printf("IN2: %d %d %d %d\n", currentElIn2.x, currentElIn2.y, currentElIn2.z, currentElIn2.w);
    printf("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\n");*/

    /*if (row < row1 && col < col2) {
        float4 currentElIn1 = (float4)(0,0,0,0); 
        float4 currentElIn2 = (float4)(0,0,0,0);

        float4 calculatedEl = (float4)(0,0,0,0); 
        for (size_t i = 0; i < col1; i++) {
            int2 coordIn1 = (int2)(row, col1); 
            int2 coordIn2 = (int2)(col1, col); 
            currentElIn1 = read_imagef(in1, coordIn1);
            currentElIn2 = read_imagef(in2, coordIn2);
            calculatedEl += currentElIn1 * currentElIn2;
        }
        write_imagef(out, coordinates, calculatedEl);
    }*/
}