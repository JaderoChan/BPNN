#ifndef BPNN_EXAMPLE_ELAPSED_TIMER_H
#define BPNN_EXAMPLE_ELAPSED_TIMER_H

#include <stdint.h>
#include <time.h>

typedef struct elapsed_timer_t
{
    clock_t start;
} elapsed_timer_t;

#define ELAPSED_TIMER_INIT {0}

void elapsed_timer_create(elapsed_timer_t* et);

void elapsed_timer_reset(elapsed_timer_t* et);

double elapsed_timer_elapsed_sec(const elapsed_timer_t* et);

double elapsed_timer_elapsed_sec_reset(elapsed_timer_t* et);

uint64_t elapsed_timer_elapsed_ms(const elapsed_timer_t* et);

uint64_t elapsed_timer_elapsed_ms_reset(elapsed_timer_t* et);

#endif // !BPNN_EXAMPLE_ELAPSED_TIMER_H
