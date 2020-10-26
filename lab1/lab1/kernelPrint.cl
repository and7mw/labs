__kernel void kernelPrint() {
    int block = get_group_id(0);
    int thread = get_local_id(0);
    int id = get_global_id(0);
    printf("I am from %d block, %d thread (global index: %d)\n", block, thread, id);
}