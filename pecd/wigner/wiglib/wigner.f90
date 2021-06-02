module wigner

  use accuracy
  use iso_c_binding
  use dffs_m
  use wigner_d
  implicit none

  contains

#include "djmk_small.f90"


  ! Computes Wigner D-function D_{m,k}^{(J)} on a three-dimensional grid of Euler angles.
  !
  ! Parameters
  ! ----------
  ! npoints : integer
  !   Number of grid points.
  ! grid(3,npoints) : c_double
  !   3D grid of different values of Euler angles, grid(1:3,ipoint) = (/phi,theta,chi/),
  !   where "phi" and "chi" are Euler angles associated with "m" and "k" quantum numbers, respectively.
  ! J : integer
  !   Value of J quantum number.
  !
  ! Returns
  ! -------
  ! val_r(npoints,2*J+1,2*J+1), val_i(npoints,2*J+1,2*J+1) : c_double
  !   Values, real and imaginary parts, of the D-function on grid, D_{m,k}^{(J)} = val(ipoint,m+J+1,k+J+1),
  !   where ipoint=1..npoints

  subroutine DJmk(J, npoints, grid, val_r, val_i) bind(c, name='DJmk')

    integer(c_int), intent(in), value :: J
    integer(c_int), intent(in), value :: npoints
    real(c_double), intent(in) :: grid(3,npoints)
    real(c_double), intent(out) :: val_r(npoints,2*J+1,2*J+1), val_i(npoints,2*J+1,2*J+1)
  
    integer(ik) :: info, m, k, ipoint, iounit
    real(rk), allocatable :: wd_matrix(:,:,:), diffwd_matrix(:,:)
    complex(rk) :: one_imag, res, val(npoints,2*J+1,2*J+1)
  
    one_imag = cmplx(0.0_rk,1.0_rk)
  
    ! initialize some data required for computing Wigner D-matrix
  
#if defined(_WIGD_FOURIER_)

    write(out, '(a)') 'DJmk: use dffs_m module for Wigner D-matrix'
    call dffs_read_coef_binary(J*2)

#elif defined(_WIGD_FOURIER_BIGJ_)

    write(out, '(a)') 'DJmk: use wigner_d module for Wigner D-matrix'
    ! allocate matrices for computing Wigner small-d matrices using module wigner_dmat2/wigner_d.f90
   allocate(wd_matrix(npoints,2*J+1,2*J+1), diffwd_matrix(2*J+1,2*J+1), stat=info)
   if (info/=0) then
      write(out, '(/a/a,10(1x,i6))') 'DJmk error: failed to allocate wd_matrix(npoints,2*J+1,2*J+1), diffwd_matrix(2*J+1,2*J+1)', 'npoints, J =', npoints, J
      stop
    endif

#else
    write(out, '(a)') 'DJmk: use slow djmk_small routine to compute Wigner D-matrix'

#endif
  
    val = 0

#if defined(_WIGD_FOURIER_BIGJ_)
    do ipoint=1, npoints
      call Wigner_dmatrix(real(j,rk), grid(2,ipoint), wd_matrix(ipoint,1:2*j+1,1:2*j+1), diffwd_matrix(1:2*j+1,1:2*j+1))
    enddo
#endif
  
    do m=-j, j
      do k=-j, j
        do ipoint=1, npoints
#if defined(_WIGD_FOURIER_)
          res = dffs( j*2, m*2, k*2, grid(2,ipoint) ) &!
              * exp( -one_imag * k * grid(3,ipoint) ) &!
              * exp( -one_imag * m * grid(1,ipoint) )
#elif defined(_WIGD_FOURIER_BIGJ_)
          res = wd_matrix(ipoint,j+m+1,j+k+1) &!
              * exp( -one_imag * k * grid(3,ipoint) ) &!
              * exp( -one_imag * m * grid(1,ipoint) )
#else
          res = djmk_small(real(j,rk), real(m,rk), real(k,rk), grid(2,ipoint)) &!
              * exp( -one_imag * k * grid(3,ipoint) ) &!
              * exp( -one_imag * m * grid(1,ipoint) )
#endif
          val(ipoint,m+j+1,k+j+1) = res
        enddo ! ipoint
      enddo ! k
    enddo ! m


#if defined(_WIGD_FOURIER_BIGJ_)
    deallocate(wd_matrix, diffwd_matrix)
#endif

    val_r = real(val, kind=rk)
    val_i = aimag(val)

  end subroutine DJmk



  ! Computes Wigner D-function D_{m,k}^{(J)} on a three-dimensional grid of Euler angles, for selected m
  !                                                                                      ----------------
  ! Parameters
  ! ----------
  ! J : integer
  !   Value of J quantum number.
  ! m : integer
  !   Value of m (-J<=m<=J) quantum number.
  ! npoints : integer
  !   Number of grid points.
  ! grid(3,npoints) : c_double
  !   3D grid of different values of Euler angles, grid(1:3,ipoint) = (/phi,theta,chi/),
  !   where "phi" and "chi" are Euler angles associated with "m" and "k" quantum numbers, respectively.
  !
  ! Returns
  ! -------
  ! val_r(npoints,2*J+1), val_i(npoints,2*J+1) : c_double
  !   Values, real and imaginary parts, of the D-function on grid, D_{m,k}^{(J)} = val(ipoint,k+J+1),
  !   where ipoint=1..npoints

  subroutine DJ_m_k(J, m, npoints, grid, val_r, val_i) bind(c, name='DJ_m_k')

    integer(c_int), intent(in), value :: J, m
    integer(c_int), intent(in), value :: npoints
    real(c_double), intent(in) :: grid(3,npoints)
    real(c_double), intent(out) :: val_r(npoints,2*J+1), val_i(npoints,2*J+1)
  
    integer(ik) :: info, k, ipoint, iounit
    real(rk), allocatable :: wd_matrix(:,:,:), diffwd_matrix(:,:)
    complex(rk) :: one_imag, res, val(npoints,2*J+1)
  
    one_imag = cmplx(0.0_rk,1.0_rk)
  
    ! initialize some data required for computing Wigner D-matrix
  
#if defined(_WIGD_FOURIER_)

    write(out, '(a)') 'DJ_m_k: use dffs_m module for Wigner D-matrix'
    call dffs_read_coef_binary(J*2)

#elif defined(_WIGD_FOURIER_BIGJ_)

    write(out, '(a)') 'DJ_m_k: use wigner_d module for Wigner D-matrix'
    ! allocate matrices for computing Wigner small-d matrices using module wigner_dmat2/wigner_d.f90
   allocate(wd_matrix(npoints,2*J+1,2*J+1), diffwd_matrix(2*J+1,2*J+1), stat=info)
   if (info/=0) then
      write(out, '(/a/a,10(1x,i6))') 'DJ_m_k error: failed to allocate wd_matrix(npoints,2*J+1,2*J+1), diffwd_matrix(2*J+1,2*J+1)', 'npoints, J =', npoints, J
      stop
    endif

#else
    write(out, '(a)') 'DJ_m_k: use slow djmk_small routine to compute Wigner D-matrix'

#endif
  
    val = 0

#if defined(_WIGD_FOURIER_BIGJ_)
    do ipoint=1, npoints
      call Wigner_dmatrix(real(j,rk), grid(2,ipoint), wd_matrix(ipoint,1:2*j+1,1:2*j+1), diffwd_matrix(1:2*j+1,1:2*j+1))
    enddo
#endif
  
    do k=-j, j
      do ipoint=1, npoints
#if defined(_WIGD_FOURIER_)
        res = dffs( j*2, m*2, k*2, grid(2,ipoint) ) &!
            * exp( -one_imag * k * grid(3,ipoint) ) &!
            * exp( -one_imag * m * grid(1,ipoint) )
#elif defined(_WIGD_FOURIER_BIGJ_)
        res = wd_matrix(ipoint,j+m+1,j+k+1) &!
            * exp( -one_imag * k * grid(3,ipoint) ) &!
            * exp( -one_imag * m * grid(1,ipoint) )
#else
        res = djmk_small(real(j,rk), real(m,rk), real(k,rk), grid(2,ipoint)) &!
            * exp( -one_imag * k * grid(3,ipoint) ) &!
            * exp( -one_imag * m * grid(1,ipoint) )
#endif
        val(ipoint,k+j+1) = res
      enddo ! ipoint
    enddo ! k


#if defined(_WIGD_FOURIER_BIGJ_)
    deallocate(wd_matrix, diffwd_matrix)
#endif

    val_r = real(val, kind=rk)
    val_i = aimag(val)

  end subroutine DJ_m_k



  ! Computes Wigner D-function d_{m,k}^{(J)}e^{-ik*chi} on a two-dimensional grid of Euler angles theta and chi, for selected m
  !                                                                                                             ----------------
  ! Parameters
  ! ----------
  ! J : integer
  !   Value of J quantum number.
  ! m : integer
  !   Value of m (-J<=m<=J) quantum number.
  ! npoints : integer
  !   Number of grid points.
  ! grid(2,npoints) : c_double
  !   2D grid of different values of Euler angles, grid(1:2,ipoint) = (/theta,chi/)
  !
  ! Returns
  ! -------
  ! val_r(npoints,2*J+1), val_i(npoints,2*J+1) : c_double
  !   Values, real and imaginary parts, of the D-function on grid, d_{m,k}^{(J)}e^{-ik*chi} = val(ipoint,k+J+1),
  !   where ipoint=1..npoints

  subroutine DJ_m_k_2D(J, m, npoints, grid, val_r, val_i) bind(c, name='DJ_m_k_2D')

    integer(c_int), intent(in), value :: J, m
    integer(c_int), intent(in), value :: npoints
    real(c_double), intent(in) :: grid(2,npoints)
    real(c_double), intent(out) :: val_r(npoints,2*J+1), val_i(npoints,2*J+1)
  
    integer(ik) :: info, k, ipoint, iounit
    real(rk), allocatable :: wd_matrix(:,:,:), diffwd_matrix(:,:)
    complex(rk) :: one_imag, res, val(npoints,2*J+1)
  
    one_imag = cmplx(0.0_rk,1.0_rk)
  
    ! initialize some data required for computing Wigner D-matrix
  
#if defined(_WIGD_FOURIER_)

    write(out, '(a)') 'DJ_m_k_2D: use dffs_m module for Wigner D-matrix'
    call dffs_read_coef_binary(J*2)

#elif defined(_WIGD_FOURIER_BIGJ_)

    write(out, '(a)') 'DJ_m_k_2D: use wigner_d module for Wigner D-matrix'
    ! allocate matrices for computing Wigner small-d matrices using module wigner_dmat2/wigner_d.f90
   allocate(wd_matrix(npoints,2*J+1,2*J+1), diffwd_matrix(2*J+1,2*J+1), stat=info)
   if (info/=0) then
      write(out, '(/a/a,10(1x,i6))') 'DJ_m_k_2D error: failed to allocate wd_matrix(npoints,2*J+1,2*J+1), diffwd_matrix(2*J+1,2*J+1)', 'npoints, J =', npoints, J
      stop
    endif

#else
    write(out, '(a)') 'DJ_m_k_2D: use slow djmk_small routine to compute Wigner D-matrix'

#endif
  
    val = 0

#if defined(_WIGD_FOURIER_BIGJ_)
    do ipoint=1, npoints
      call Wigner_dmatrix(real(j,rk), grid(1,ipoint), wd_matrix(ipoint,1:2*j+1,1:2*j+1), diffwd_matrix(1:2*j+1,1:2*j+1))
    enddo
#endif
  
    do k=-j, j
      do ipoint=1, npoints
#if defined(_WIGD_FOURIER_)
        res = dffs( j*2, m*2, k*2, grid(1,ipoint) ) &!
            * exp( -one_imag * k * grid(2,ipoint) )
#elif defined(_WIGD_FOURIER_BIGJ_)
        res = wd_matrix(ipoint,j+m+1,j+k+1) &!
            * exp( -one_imag * k * grid(2,ipoint) )
#else
        res = djmk_small(real(j,rk), real(m,rk), real(k,rk), grid(1,ipoint)) &!
            * exp( -one_imag * k * grid(2,ipoint) )
#endif
        val(ipoint,k+j+1) = res
      enddo ! ipoint
    enddo ! k


#if defined(_WIGD_FOURIER_BIGJ_)
    deallocate(wd_matrix, diffwd_matrix)
#endif

    val_r = real(val, kind=rk)
    val_i = aimag(val)

  end subroutine DJ_m_k_2D

end module wigner