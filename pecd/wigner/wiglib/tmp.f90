program tmp

  ! to convert textual from of coefficients in dffs_m module into binary format

  use dffs_m
  implicit none

  call dffs_read_coef(100*2)
  call dffs_write_coef_binary

end program tmp