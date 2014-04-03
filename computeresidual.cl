#include "constants.h"

__kernel void ComputeResidual(__global const REAL* restrict prim, 
                              __global const REAL* restrict dprim_dt,
                              __global REAL* restrict F)
{

    int i = get_global_id(0);
    int j = get_global_id(1);
    int iTile = get_local_id(0);
    int jTile = get_local_id(1);    

    // Tile of type __local memory which needs to fit in the cache.
    __local REAL primTile[(TILE_SIZE_X1+2*NG)*(TILE_SIZE_X2+2*NG)*DOF];

    // Fill in the core of the tile.
    for (int var=0; var<DOF; var++) {
        primTile[INDEX_LOCAL(iTile,jTile,var)] = prim[INDEX_GLOBAL(i,j,var)];
    }

    // Fill in the boundaries of the tile. If the edge of the tile coincides
    // with the edge of the domain, then apply boundary conditions to the tile.
    // Here we apply periodic boundary conditions.
    for (int var=0; var<DOF; var++) {

        if (iTile==0) {
            if (i>=NG) {
            // Tile not on the left edge of the global domain. Copy data from
            // the global array.
                for (int iNg=-NG; iNg<0; iNg++) {
                    primTile[INDEX_LOCAL(iNg,jTile,var)] =
                        prim[INDEX_GLOBAL(i+iNg,j,var)];
                }
            } else {
            // Tile on the left edge of the global domain. Apply periodic
            // boundary conditions.
                for (int iNg=-NG; iNg<0; iNg++) {
                    primTile[INDEX_LOCAL(iNg,jTile,var)] =
                        prim[INDEX_GLOBAL(N1+iNg,j,var)];
                }
            }
        }
    
        if (iTile==TILE_SIZE_X1-1) {
            if (i<=N1-NG) {
            // Tile not on the right edge of the global domain. Copy data from
            // the global array.
                for (int iNg=0; iNg<NG; iNg++) {
                    primTile[INDEX_LOCAL(TILE_SIZE_X1+iNg,jTile,var)] =
                        prim[INDEX_GLOBAL(i+iNg+1,j,var)];
                }
            } else {
            // Tile on the right edge of the global domain. Apply periodic
            // boundary conditions.
                for (int iNg=0; iNg<NG; iNg++) {
                    primTile[INDEX_LOCAL(TILE_SIZE_X1+iNg,jTile,var)] =
                        prim[INDEX_GLOBAL(iNg,j,var)];
                }
            }
        }
       
        if (jTile==0) {
            if (j>=NG) {
            // Tile not on the bottom edge of the global domain. Copy data from
            // the global array.
                for (int jNg=-NG; jNg<0; jNg++) {
                    primTile[INDEX_LOCAL(iTile,jNg,var)] =
                        prim[INDEX_GLOBAL(i,j+jNg,var)];
                }
            } else {
            // Tile on the bottom edge of the global domain. Apply periodic
            // boundary conditions.
                for (int jNg=-NG; jNg<0; jNg++) {
                    primTile[INDEX_LOCAL(iTile,jNg,var)] =
                        prim[INDEX_GLOBAL(i,N2+jNg,var)];
                }
            }
        }
    
        if (jTile==TILE_SIZE_X2-1) {
            if (j<=N2-NG) {
            // Tile not on the top edge of the global domain. Copy data from
            // the global array.
                for (int jNg=0; jNg<NG; jNg++) {
                    primTile[INDEX_LOCAL(iTile,TILE_SIZE_X2+jNg,var)] =
                        prim[INDEX_GLOBAL(i,j+jNg+1,var)];
                }
            } else {
            // Tile on the top edge of the global domain. Apply periodic
            // boundary conditions.
                for (int jNg=0; jNg<NG; jNg++) {
                    primTile[INDEX_LOCAL(iTile,TILE_SIZE_X2+jNg,var)] =
                        prim[INDEX_GLOBAL(i,jNg,var)];
                }
            }
        }
    
    }

    // All the threads need to synchronize before proceeding.
    barrier(CLK_LOCAL_MEM_FENCE);

    
    // Now do all the needed computation.

    for (int var=0; var<DOF; var++) {
        F[INDEX_GLOBAL(i,j,var)] = dprim_dt[INDEX_GLOBAL(i,j,var)] -
                                  (primTile[INDEX_LOCAL(iTile+1,jTile,var)] -
                                   primTile[INDEX_LOCAL(iTile,jTile,var)])/DX1 -
                                  (primTile[INDEX_LOCAL(iTile,jTile+1,var)] -
                                   primTile[INDEX_LOCAL(iTile,jTile,var)])/DX2;

    }

}

