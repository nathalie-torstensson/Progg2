""" MA3.py

Student:
Mail:
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
from numba import njit
import functools as f

# Exc1
#metoden är byggd på slumpmässigheter
#förhållandet mellan antal prickar innanför och untanför cirkeln
#är ungefär lika med cirkelns area/ kvadratens area
#detta ger pi = 4 *(nc/n)
#mer prickar innebär bättre uppsakattning
def approximate_pi(n):
    # n is the number of points
    #lista för punkter innanför och utanför cirekeln
    inside_x, inside_y = [], [] #punkter inuti cirkeln
    outside_x, outside_y = [], [] #punkter utanför cirkeln
    # skapa n slumpmässiga punkter
    for _ in range(n):
        #genererar en slumpkordinat
         x = random.uniform(-1,1)
         y = random.uniform(-1, 1)

         #avståndet beräknat med pytagoras
         if x**2 + y**2 <= 1: #pytagoras sats, för om den är mindre kommer det vara innanför cirkeln, då mitt i origo
            inside_x.append(x)
            inside_y.append(y)

         else:
             outside_x.append(x)
             outside_y.append(y)

    n_c = len(inside_x)

    #formeln för monte carlo är cirkelns area/kvadratens area = pi/4
    #pi = 4*(n_c/n)
    pi_approximate = 4* n_c / n
    print(f'Amount of data points (n): {n}')
    print(f'Approximate of pi: {pi_approximate}')

    plt.scatter(inside_x, inside_y, color = 'red')
    plt.scatter(outside_x, outside_y, color = 'blue') 
    # Write your code here
    return pi_approximate

# Exc2, approximation
def sphere_volume(n, d): 
    # n is the number of points
    #den inre delen av listan skapar EN punkt med d koordinater
    #den yttre looparen upprepar detta n gånger
    points = [[random.uniform(-1,1) for _ in range(d)] for _ in range(n)]
    #skapar d punkter, den yttre delen upprepar det n gånger

    def inside(points):
           #map används för att tillämpa detta på varje värde i min lista
        squared_coordinates = map(lambda x: x**2, points) #kvadreerar alla punkter
        squared_sum = f.reduce(lambda a, b: a + b, squared_coordinates)

        return squared_sum <= 1

#tar bort alla punkter ur listan som returnerar false och därmed ej är inside
    inside_points = list(filter(inside, points))
        
    # d is the number of dimensions of the sphere 
    #från -1 till1 ger en längd på 2

    #2 då det är från -1 till 1
    cube = 2**d
    #Antalet punkter i sfären/ alla punkter detta gånger kubens volym
    return (len(inside_points)/n) * cube

#Exc2, real value
def hypersphere_exact(n, d):
    # n is the number of points
    return (m.pi **(d/2)) / (m.gamma(d/2+1))
    # d is the number of dimensions of the sphere 
    

#Exc3: numba version
@njit
def sphere_volume_numba(n:int, d:int)->float:
    # n is the number of points
    inside_count = 0
    #genererar n stycken dara punkre
    for _ in range(n):
        squared_sum = 0
        #beräknar kvadratsumman för alla dimensioner
        for _ in range(d):
            x = random.uniform(-1.0, 1.0)
            squared_sum += x*x

        #om det är innan eller utanför
        if squared_sum <= 1:
            inside_count +=1

    cube = 2.0**d

    # d is the number of dimensions of the sphere
    #np is the number of processes
    return (inside_count/n) * cube

def count_inside_parts(n_parts, d):
    #hjälpfunktion som delar upp i delar för att använda vid parallel processing
    inside_count = 0
    for _ in range(n_parts): #delar upp n i större delar för att räkna på flera kärnor
        squared_sum = sum(random.uniform(-1.0, 1.0) **2 for _ in range(d))
        if squared_sum <= 1.0:
            inside_count +=1
    return inside_count


#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallel(n, d, np=10):
    # n is the number of points
    parts_size = n//np # delar upp beräkningarna i delar
    parts = [parts_size] *np #skapar en lista med alla objekt
    parts[-1] += n%np #om n inte var jämnt hamnar de sista punkterna på den sista beräkningen

    #skapar np olika separata pytonprocesser
    with future.ProcessPoolExecutor(max_workers=np) as executor:
        #skickar iväg varje count_inside_parts till en separat cpu
        futures =[executor.submit(count_inside_parts, part, d)for part in parts]

        #tar summan av alla processer och vad dem får för värde
        total = sum(f.result() for f in futures)

    cube = 2.0**d

    return (total/n)* cube
    # d is the number of dimensions of the sphere
    # np is the number of processes

    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # Exc3
    n = 1000000
    d = 11
    for i in range(1,4):
        start = pc()
        res= sphere_volume(n, d)
        stop = pc()
        print(f"Exc3: Sequential time of {d} and {n}: {stop-start}")
    
    for i in range(1,4):
        start= pc()
        res = sphere_volume_numba(n,d)
        stop = pc()
        print(f"What is numba time?: {stop-start:.5f}")

    # Exc4
    n = 1000000
    d = 11
    np_processes = 10
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")

    start = pc()
    sphere_volume_parallel(n,d, np = np_processes)
    stop = pc()
    print(f"What is parallel time? ({np_processes} processes) of d={d} and n={n}: {stop-start}")

    
    

if __name__ == '__main__':
	main()


'''
3.b) Exc3: Sequential time of 11 and 1000000: 2.107629916165024
Exc3: Sequential time of 11 and 1000000: 2.0394515837542713
Exc3: Sequential time of 11 and 1000000: 2.0707429591566324
What is numba time?: 0.84072
What is numba time?: 0.01772
What is numba time?: 0.01776
Numba tar längre tid första gången då den översätter min pyton kod till 
c maskinkod, Numba använder sedan den färdiga maskinkoden och bygger inte nya listor.

Det som gör att det tar längre tid är JIT komplieringen, sen är de maskin
3.c)Pyton skapar en lång lista med flyttal, numba använder istället variablar
enkla for loopar ger helt enkel snabbare resultat
'''