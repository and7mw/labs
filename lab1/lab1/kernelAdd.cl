__kernel void kernelAdd(__global int *buffer, const unsigned int count) {
    unsigned int id = get_global_id(0);

    if (id < count)
        buffer[id] += id;
}
