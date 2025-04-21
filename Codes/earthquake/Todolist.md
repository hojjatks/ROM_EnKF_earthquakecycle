Deadline: Apr 30 to finsih paper 
----------------------------------------
plan: 
----------------------------------------
thought processes:
** on the hyper parameter I chose, I picked Ntout=1500 I want to pick as large as number as possible to make longer forecasts and get rid of chaos more effectively, but as you increase this you dont have a good temporal resolution for the interevent period. Some important thing that I learned is that it is good to choose Nt as large as possible. But if you choose it too large, because the system is chaotic the ML model cannot learn such models. So choose a large Ntout, if you cannot learn g_1, decrease it to a smaller number. It is very intersting. When Nt is large the output of the neural network has a lot of spikes which makes the learning very challenging. I decreaased it by a factor of 5 to see what happens. There are still spikes but I tried to downsample the data and remove some  of the data during the event period. Still I think there is noise in the system. By noise, I mean the system is too nonlinear and you cannot not take large time steps and you cannot learn the noise. As a reuslt, I think you should decrease that to reduce that "noise". Please increase Nxout as well. You don't need that much resolution.
** I am thinking of having more samples outside the chaotic attractor and farther with shorter times, this will help to learn regions outside the chaotic attractor more effectively.
** I think 5500 yearrs with 100 events is not enough. How about twice this size to find the POD modes?

-------------------------------------------------------------------------------------------
Finished Jobs:
** picked proper model parameters to simulate. I think I picked drs=0.012 which makes chaotic behaviour and also it is a case which is not too small and not too big. The number of required elements in this case is the same as the case for SSEs.
** Clean the forward model and run the program with Nx=1 for all nucleation size and calculate the POD components for different values. You are going to use the POD modes in generating initial conditions. You are also going to use the code to simulate the initial conditions. YOU need a very clean code to simulate the fault. 



-------------------------------------------------------------------------------------------
Currently focusing:
** constantly use github



earthquake:


** plot the eigenfunctions and the coseismic slip for this drs.
** learn the function $$g_1$$


SSEs:


** plot MFD for the ROM.

-------------------------------------------------------------------------------------------
To do list for earthquakes:

* pick proper model parameters to simulate.
* update the results of paleoseismic data because you changed the your base simulation (change figure 8 and figure 9)
* simulate the fault 100 times for a long time. (1 day)
* add some points directing towards the center.
* write ML codes to learn g_1 and g_2.
* check the statistics of the events in the ROM and the orginal problem.
* Run the EnKF 
* Plot the prediction performance.

To do list for SSEs:
* plot MFD for the ROM.
