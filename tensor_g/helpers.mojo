fn __check_bounds(index: Int, size: Int):
    let index_in_bounds = index >= 0 and index < size
    debug_assert(index_in_bounds, "Error index out of bounds.")


fn __negative_pos_to_positive(index: Int, size: Int) -> Int:
    if index < 0:
        return size + index
    return index
