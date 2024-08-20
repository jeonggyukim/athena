/* -----------------------------------------------------------------------------
 *
 *  Copyright (C) 1997-2012 Krzysztof M. Gorski, Eric Hivon, Martin Reinecke,
 *                          Benjamin D. Wandelt, Anthony J. Banday,
 *                          Matthias Bartelmann,
 *                          Reza Ansari & Kenneth M. Ganga
 *
 *
 *  Original source code from HEALPix.
 *
 *  HEALPix is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  HEALPix is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with HEALPix; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *  For more information about HEALPix see http://healpix.sourceforge.net
 *
 *---------------------------------------------------------------------------*/

// Athena++ headers
#include "../athena.hpp"

static const Real halfpi=1.570796326794896619231321691639751442099;

static const short ctab[]={
  0,1,256,257,2,3,258,259,512,513,768,769,514,515,770,771,4,5,260,261,6,7,262,
  263,516,517,772,773,518,519,774,775,1024,1025,1280,1281,1026,1027,1282,1283,
  1536,1537,1792,1793,1538,1539,1794,1795,1028,1029,1284,1285,1030,1031,1286,
  1287,1540,1541,1796,1797,1542,1543,1798,1799,8,9,264,265,10,11,266,267,520,
  521,776,777,522,523,778,779,12,13,268,269,14,15,270,271,524,525,780,781,526,
  527,782,783,1032,1033,1288,1289,1034,1035,1290,1291,1544,1545,1800,1801,1546,
  1547,1802,1803,1036,1037,1292,1293,1038,1039,1294,1295,1548,1549,1804,1805,
  1550,1551,1806,1807,2048,2049,2304,2305,2050,2051,2306,2307,2560,2561,2816,
  2817,2562,2563,2818,2819,2052,2053,2308,2309,2054,2055,2310,2311,2564,2565,
  2820,2821,2566,2567,2822,2823,3072,3073,3328,3329,3074,3075,3330,3331,3584,
  3585,3840,3841,3586,3587,3842,3843,3076,3077,3332,3333,3078,3079,3334,3335,
  3588,3589,3844,3845,3590,3591,3846,3847,2056,2057,2312,2313,2058,2059,2314,
  2315,2568,2569,2824,2825,2570,2571,2826,2827,2060,2061,2316,2317,2062,2063,
  2318,2319,2572,2573,2828,2829,2574,2575,2830,2831,3080,3081,3336,3337,3082,
  3083,3338,3339,3592,3593,3848,3849,3594,3595,3850,3851,3084,3085,3340,3341,
  3086,3087,3342,3343,3596,3597,3852,3853,3598,3599,3854,3855 };

static const int jrll[] = { 2,2,2,2,3,3,3,3,4,4,4,4 };
static const int jpll[] = { 1,3,5,7,0,2,4,6,1,3,5,7 };

static void nest2xyf (int nside, int pix, int *ix, int *iy, int *face_num) {
  int npface_=nside*nside, raw;
  *face_num = pix/npface_;
  pix &= (npface_-1);
  raw = (pix&0x5555) | ((pix&0x55550000)>>15);
  *ix = ctab[raw&0xff] | (ctab[raw>>8]<<4);
  pix >>= 1;
  raw = (pix&0x5555) | ((pix&0x55550000)>>15);
  *iy = ctab[raw&0xff] | (ctab[raw>>8]<<4);
}

void pix2ang_nest_z_phi(int nside_, int pix, Real *z, Real *phi) {
  int nl4 = nside_*4;
  int npix_=12*nside_*nside_;
  Real fact2_ = 4./npix_;
  int face_num, ix, iy, jr, nr, kshift, jp;

  nest2xyf(nside_,pix,&ix,&iy,&face_num);
  jr = (jrll[face_num]*nside_) - ix - iy - 1;

  if (jr<nside_)
    {
    nr = jr;
    *z = 1 - nr*nr*fact2_;
    kshift = 0;
    }
  else if (jr > 3*nside_)
    {
    nr = nl4-jr;
    *z = nr*nr*fact2_ - 1;
    kshift = 0;
    }
  else
    {
    Real fact1_ = (nside_<<1)*fact2_;
    nr = nside_;
    *z = (2*nside_-jr)*fact1_;
    kshift = (jr-nside_)&1;
    }

  jp = (jpll[face_num]*nr + ix -iy + 1 + kshift) / 2;
  if (jp>nl4) jp-=nl4;
  if (jp<1) jp+=nl4;

  *phi = (jp-(kshift+1)*0.5)*(halfpi/nr);
}

/* PUBLIC FUNCTION */
/*! Sets \a vec to the Cartesian vector pointing in the direction of the center
    of pixel \a ipix in NEST scheme at resolution \a nside. */
//
// NOTE: This public function is re-implemented in HEALPixWrapper class

// void pix2vec_nest(int nside, int ipix, Real *vec) {
//   Real z, phi, stheta;
//   pix2ang_nest_z_phi(nside,ipix,&z,&phi);
//   stheta=std::sqrt((1.-z)*(1.+z));
//   vec[0]=stheta*std::cos(phi);
//   vec[1]=stheta*std::sin(phi);
//   vec[2]=z;
// }
