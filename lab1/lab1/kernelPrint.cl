__kernel void kernelPrint() {
    int block = get_group_id(0);
    int thread = get_local_id(0);
    int id = get_global_id(0);
    printf("I am from %d block, %d thread (global index: %d)\n", block, thread, id);
}
printf("GROUP: %d and %d", get_group_id(0), get_group_id(1));
        for (int row = 0; row < BLOCK_SIZE; row++) {
            for (int col = 0; col < BLOCK_SIZE; col++) {
                std::cout << in1[row * col2 + col] << " ";
            }
            std::cout << std::endl;
        }