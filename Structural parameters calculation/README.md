# How to use
1. Paste cluster coordinations in line 192

2. Input uncertainty in line 228, calculated by sqrt(A^2+B^2+C^2)

3. Run the scripts

4. Type the indexs, connected by '-'. If using faces or centriods, using ','. For example:
   
   4.1 0-1 (distance between atom 0 and atom 1)

   4.2 0,1,2,3,4-5 (distance between centriod of atoms 0,1,2,3,4 and atom 5)

   4.3 0-1-2 (angle of atoms 0-1-2)

   4.4 f0,1,2,3,4-5 (distance between fitted plane of atoms 0,1,2,3,4 and atom 5)

   4.5 f0,1,2,3,4-f5,6,7,8,9 (angle of fitted plane of atoms 0,1,2,3,4 and fitted plane of atoms 5,6,7,8,9)
