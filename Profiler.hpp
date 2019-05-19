#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <chrono>

class Profiler
{
public:

	typedef std::chrono::time_point<std::chrono::high_resolution_clock> Timepoint;
	typedef std::chrono::duration<double, std::milli> Duration;

	static Timepoint now()
	{
		return std::chrono::high_resolution_clock::now();
	}

	static long Elapsed(Timepoint start)
	{
		Duration elapsed = now() - start;

		return (long)elapsed.count();
	}
};

#endif
