#define N1 64
#define N2 64
#define X1_MIN 0.
#define X1_MAX 1.
#define X2_MIN 0.
#define X2_MAX 1.
#define NG 2
#define DOF 2

#define REAL double
#define TILE_SIZE_X1 8
#define TILE_SIZE_X2 8
#define DX1 ((X1_MAX-X1_MIN)/(REAL)N1)
#define DX2 ((X2_MAX-X2_MIN)/(REAL)N2)
// Index of the global array
#define INDEX_GLOBAL(i,j,var) (var + DOF*(i + N1*(j)))
// Index inside a tile
#define INDEX_LOCAL(iTile,jTile,var) (iTile+NG + \
                                      (TILE_SIZE_X1+2*NG)*(jTile+NG + \
                                      (TILE_SIZE_X2+2*NG)*(var)))
