#ifndef APPS_UTILS_H
#define APPS_UTILS_H

#include "include_all.h"

uint64_t now();

inline void really_now(volatile uint64_t *ptr)
{
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    *ptr = ts.tv_sec*1'000'000'000 + ts.tv_nsec;
    __asm__ __volatile__("mfence");
}

#endif

