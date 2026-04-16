#include <cstdlib>
#include <iostream>
#include <random>

#include "functions.h"

void initialize_random_vals(float *arr, int N)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 99);

    for (int i = 0; i < N; i++)
    {
        arr[i] = dist(gen);
    }
}

void print_array_vals(float *arr, int N)
{
    for (int i = 0; i < N; ++i)
    {
        std::cout << arr[i] << " ";
    }

    std::cout << std::endl;
}