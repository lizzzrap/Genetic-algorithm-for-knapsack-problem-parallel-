#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <ctime>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <omp.h>


using namespace std;


const string testPath = "./data/";
const string testFiles[] = {
    "ks_4_0",
    "ks_19_0",
    "ks_30_0",
    "ks_40_0",
    "ks_45_0",
    "ks_50_0",
    "ks_50_1",
    "ks_60_0",
    "ks_100_0",
    "ks_100_1",
    "ks_100_2",
    "ks_200_0",
    "ks_200_1",
    "ks_300_0",
    "ks_400_0",
    "ks_500_0",
    "ks_1000_0",
    "ks_10000_0",
};


struct Item {
    int value;
    int weight;
};

class GeneticKnapsack {
private:
    int capacity;
    vector<Item> items;
    int populationSize;
    double mutationRate;
    int maxGenerations;

public:
    GeneticKnapsack(int cap, const vector<Item>& its, int popSize = 100,
        double mutRate = 0.01, int maxGen = 1000)
        : capacity(cap), items(its), populationSize(popSize),
        mutationRate(mutRate), maxGenerations(maxGen) {
    }

    vector<bool> generateRandomIndividual() {
        vector<bool> individual(items.size());
        static mt19937 gen(time(nullptr));
        //uniform_int_distribution<int> dist(0, 1);
        //for (size_t i = 0; i < items.size(); ++i) {
        //    individual[i] = dist(gen);
        //}

        uniform_int_distribution<int> dist(0, items.size() - 1);
        int capacityIndividual = 0;
        for (size_t i = 0; i < items.size() && capacityIndividual < capacity; ++i) {
            size_t j = dist(gen);
            if (!individual[j]) {
                if (capacity >= capacityIndividual + items[j].weight) {
                    capacityIndividual += items[j].weight;
                    individual[j] = true;
                }
            }
        }

        return individual;
    }

    int calculateFitness(const vector<bool>& individual) {
        int totalValue = 0;
        int totalWeight = 0;

        for (size_t i = 0; i < individual.size(); ++i) {
            if (individual[i]) {
                totalValue += items[i].value;
                totalWeight += items[i].weight;
            }
        }

        if (totalWeight > capacity) {
            return 0;
        }

        return totalValue;
    }

    vector<bool> tournamentSelection(const vector<vector<bool>>& population,
        const vector<int>& fitness) {
        static mt19937 gen(time(nullptr));
        uniform_int_distribution<int> dist(0, population.size() - 1);

        int a = dist(gen);
        int b = dist(gen);

        return fitness[a] > fitness[b] ? population[a] : population[b];
    }

    pair<vector<bool>, vector<bool>> crossover(const vector<bool>& parent1,
        const vector<bool>& parent2) {
        static mt19937 gen(time(nullptr));
        uniform_int_distribution<int> dist(0, parent1.size() - 1);
        int crossoverPoint = dist(gen);

        vector<bool> child1 = parent1;
        vector<bool> child2 = parent2;

        for (int i = crossoverPoint; i < parent1.size(); ++i) {
            swap(child1[i], child2[i]);
        }

        return { child1, child2 };
    }

    void mutate(vector<bool>& individual) {
        static mt19937 gen(time(nullptr));
        uniform_real_distribution<double> dist(0.0, 1.0);

        for (size_t i = 0; i < individual.size(); ++i) {
            if (dist(gen) < mutationRate) {
                individual[i] = !individual[i];
            }
        }
    }

    pair<vector<bool>, int> solve() {
        vector<vector<bool>> population(populationSize);
        vector<int> fitness(populationSize);

        for (int i = 0; i < populationSize; ++i) {
            population[i] = generateRandomIndividual();
            fitness[i] = calculateFitness(population[i]);
        }

        for (int generation = 0; generation < maxGenerations; ++generation) {
            vector<vector<bool>> newPopulation;

            auto bestIt = max_element(fitness.begin(), fitness.end());
            int bestIdx = distance(fitness.begin(), bestIt);
            newPopulation.push_back(population[bestIdx]);

            while (newPopulation.size() < populationSize) {
                auto parent1 = tournamentSelection(population, fitness);
                auto parent2 = tournamentSelection(population, fitness);
                auto child = crossover(parent1, parent2);

                mutate(child.first);
                mutate(child.second);

                newPopulation.push_back(child.first);
                if (newPopulation.size() < populationSize) {
                    newPopulation.push_back(child.second);
                }
            }

            population = newPopulation;

            for (int i = 0; i < populationSize; ++i) {
                fitness[i] = calculateFitness(population[i]);
            }

            /*if (generation % 100 == 0) {
                cout << "Generation " << generation << ": Best fitness = "
                    << *max_element(fitness.begin(), fitness.end()) << endl;
            }*/
        }
        auto bestIt = max_element(fitness.begin(), fitness.end());
        int bestIdx = distance(fitness.begin(), bestIt);
        return { population[bestIdx], fitness[bestIdx] };
    }
};

class ParallelGeneticKnapsack {
private:
    int capacity;
    vector<Item> items;
    int populationSize;
    double mutationRate;
    int maxGenerations;
    int numThreads;

public:
    ParallelGeneticKnapsack(int cap, const vector<Item>& its, int popSize = 100,
        double mutRate = 0.01, int maxGen = 1000, int threads = 4)
        : capacity(cap), items(its), populationSize(popSize),
        mutationRate(mutRate), maxGenerations(maxGen), numThreads(threads) {
    }

    vector<bool> generateRandomIndividual(mt19937& gen) {
        vector<bool> individual(items.size());
        //uniform_int_distribution<int> dist(0, 1);
        //for (size_t i = 0; i < items.size(); ++i) {
        //    individual[i] = dist(gen);
        //}

        uniform_int_distribution<int> dist(0, items.size() - 1);
        int capacityIndividual = 0;
        for (size_t i = 0; i < items.size() && capacityIndividual < capacity; ++i) {
            size_t j = dist(gen);
            if (!individual[j]) {
                if (capacity >= capacityIndividual + items[j].weight) {
                    capacityIndividual += items[j].weight;
                    individual[j] = true;
                }
            }
        }

        return individual;
    }

    void initializePopulation(vector<vector<bool>>& population, vector<int>& fitness) {
#pragma omp parallel num_threads(numThreads)
        {
            mt19937 gen(time(nullptr) + omp_get_thread_num());

#pragma omp for
            for (int i = 0; i < populationSize; ++i) {
                population[i] = generateRandomIndividual(gen);
                fitness[i] = calculateFitness(population[i]);
            }
        }
    }

    int calculateFitness(const vector<bool>& individual) {
        int totalValue = 0;
        int totalWeight = 0;

        for (size_t i = 0; i < individual.size(); ++i) {
            if (individual[i]) {
                totalValue += items[i].value;
                totalWeight += items[i].weight;
            }
        }

        return (totalWeight > capacity) ? 0 : totalValue;
    }

    vector<bool> tournamentSelection(const vector<vector<bool>>& population,
        const vector<int>& fitness, mt19937& gen) {
        uniform_int_distribution<int> dist(0, population.size() - 1);

        int a = dist(gen);
        int b = dist(gen);

        return fitness[a] > fitness[b] ? population[a] : population[b];
    }

    pair<vector<bool>, vector<bool>> crossover(const vector<bool>& parent1,
        const vector<bool>& parent2, mt19937& gen) {
        uniform_int_distribution<int> dist(0, parent1.size() - 1);
        int crossoverPoint = dist(gen);

        vector<bool> child1 = parent1;
        vector<bool> child2 = parent2;

        for (int i = crossoverPoint; i < parent1.size(); ++i) {
            swap(child1[i], child2[i]);
        }

        return { child1, child2 };
    }

    void mutate(vector<bool>& individual, mt19937& gen) {
        uniform_real_distribution<double> dist(0.0, 1.0);

        for (size_t i = 0; i < individual.size(); ++i) {
            if (dist(gen) < mutationRate) {
                individual[i] = !individual[i];
            }
        }
    }

    pair<vector<bool>, int> solve() {
        vector<vector<bool>> population(populationSize);
        vector<int> fitness(populationSize);

        initializePopulation(population, fitness);

        for (int generation = 0; generation < maxGenerations; ++generation) {
            vector<vector<bool>> newPopulation;

            auto bestIt = max_element(fitness.begin(), fitness.end());
            int bestIdx = distance(fitness.begin(), bestIt);
            newPopulation.push_back(population[bestIdx]);

#pragma omp parallel num_threads(numThreads)
            {
                mt19937 gen(time(nullptr) + omp_get_thread_num());
                vector<vector<bool>> localNewPopulation;

#pragma omp for nowait
                for (int i = 1; i < populationSize; i += 2) {
                    if (newPopulation.size() >= populationSize) break;

                    auto parent1 = tournamentSelection(population, fitness, gen);
                    auto parent2 = tournamentSelection(population, fitness, gen);

                    auto child = crossover(parent1, parent2, gen);

                    mutate(child.first, gen);
                    mutate(child.second, gen);

                    localNewPopulation.push_back(child.first);
                    if (i + 1 < populationSize) {
                        localNewPopulation.push_back(child.second);
                    }
                }

#pragma omp critical
                {
                    newPopulation.insert(newPopulation.end(),
                        localNewPopulation.begin(),
                        localNewPopulation.end());
                }
            }

            if (newPopulation.size() > populationSize) {
                newPopulation.resize(populationSize);
            }

            population = newPopulation;

#pragma omp parallel for num_threads(numThreads)
            for (int i = 0; i < populationSize; ++i) {
                fitness[i] = calculateFitness(population[i]);
            }

            //if (generation % 100 == 0) {
            //    cout << "Generation " << generation << ": Best fitness = "
            //        << *max_element(fitness.begin(), fitness.end())
            //        << " (Threads: " << omp_get_max_threads() << ")" << endl;
            //}
        }

        auto bestIt = max_element(fitness.begin(), fitness.end());
        int bestIdx = distance(fitness.begin(), bestIt);
        return { population[bestIdx], fitness[bestIdx] };
    }
};


int main() {
    int populationSize = 400;
    int maxGenerations = 500;
    double mutationRate = 0.1;
    int numThreads = 3; // omp_get_max_threads();

    cout << "The solution Knapsack problem with genetic algorithm." <<  endl;
    cout << "populationSize: " << populationSize << ", " << "maxGenerations: " << maxGenerations << ", " << "numThreads: " << numThreads << endl << endl;
    cout << setw(10) << "TestFile" << setw(30) << "Maximum value (single thread)" << setw(30) << "Maximum value (multithreads)" << endl;

    for (string testFile : testFiles)
    {
        cout << setw(10) << testFile;

        int n, capacity;
        vector<Item> items;

        ifstream in(testPath + testFile); // окрываем файл для чтения
        if (in.is_open()) {
            in >> n >> capacity;
            items.resize(n);
            for (int i = 0; i < n; ++i) {
                in >> items[i].value >> items[i].weight;
            }
            in.close();
        }
        else {
            cout << endl << "File not found." << endl;
            return -1;
        }


        using chrono::high_resolution_clock;
        using chrono::duration_cast;
        using chrono::nanoseconds;


        auto t1 = high_resolution_clock::now();
        GeneticKnapsack solver(capacity, items, populationSize, mutationRate, maxGenerations);
        auto solution = solver.solve();
        auto t2 = high_resolution_clock::now();
        auto ns = duration_cast<nanoseconds>(t2 - t1);
        cout << setw(13) << solution.second << " / " << setw(12) << ns.count() << "ns";

        t1 = high_resolution_clock::now();
        ParallelGeneticKnapsack solverP(capacity, items, populationSize, mutationRate, maxGenerations, numThreads);
        solution = solverP.solve();
        t2 = high_resolution_clock::now();
        auto nsP = duration_cast<nanoseconds>(t2 - t1);
        int percent = 100 * nsP / ns;
        cout << setw(13) << solution.second << " / " << setw(12) << nsP.count() << "ns" << " (" << percent << "%)" << endl;



        /*cout << "Selected items (1 - selected, 0 - not selected): ";
        for (bool b : solution.first) {
            cout << b << " ";
        }
        cout << endl;*/

        // Проверка веса
        /*int totalWeight = 0;
        for (size_t i = 0; i < solution.first.size(); ++i) {
            if (solution.first[i]) {
                totalWeight += items[i].weight;
            }
        }
        cout << "Total weight: " << totalWeight << " (capacity: " << capacity << ")" << endl << endl;*/

    }

    return 0;
}