#include "elapsed_timer.h"

void elapsed_timer_create(elapsed_timer_t* et)
{
    if (!et) return;
    et->start = clock();
}

void elapsed_timer_reset(elapsed_timer_t* et)
{
    if (!et) return;
    et->start = clock();
}

double elapsed_timer_elapsed_sec(const elapsed_timer_t* et)
{
    if (!et) return -1.0;
    return (double) (clock() - et->start) / (double) CLOCKS_PER_SEC;
}

double elapsed_timer_elapsed_sec_reset(elapsed_timer_t* et)
{
    if (!et) return -1.0;
    const double e = (double) (clock() - et->start) / (double) CLOCKS_PER_SEC;
    et->start = clock();
    return e;
}

uint64_t elapsed_timer_elapsed_ms(const elapsed_timer_t* et)
{
    const double e = elapsed_timer_elapsed_sec(et);
    return (e < 0 ? (uint64_t) -1 : (uint64_t) (e * 1000));
}

uint64_t elapsed_timer_elapsed_ms_reset(elapsed_timer_t* et)
{
    const double e = elapsed_timer_elapsed_sec_reset(et);
    return (e < 0 ? (uint64_t) -1 : (uint64_t) (e * 1000));
}
