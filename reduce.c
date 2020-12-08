#include <stdio.h>
#include "mpi.h"

main(int argc, char* argv[]){
   int np, pid, dest, source, tag = 0;
   float data, tmp;
   MPI_Status status;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &np);
   MPI_Comm_rank(MPI_COMM_WORLD, &pid);

   data = pid + 100.0;

   if(pid!=0){
      MPI_Send(&data, 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
   }
   else{
      for(dest=1;dest<np;dest++){
         tmp = data;
         MPI_Recv(&data, 1, MPI_FLOAT, dest, tag, MPI_COMM_WORLD, &status);
         data += tmp;
      }
   }

   if (pid == 0) printf("%f\n", data);

   MPI_Finalize();
}

