ERROR: ld.so: object 'libdlfaker.so' from LD_PRELOAD cannot be preloaded: ignored.
ERROR: ld.so: object 'libvglfaker.so' from LD_PRELOAD cannot be preloaded: ignored.
ERROR: ld.so: object 'libdlfaker.so' from LD_PRELOAD cannot be preloaded: ignored.
ERROR: ld.so: object 'libvglfaker.so' from LD_PRELOAD cannot be preloaded: ignored.
ERROR: ld.so: object 'libdlfaker.so' from LD_PRELOAD cannot be preloaded: ignored.
ERROR: ld.so: object 'libvglfaker.so' from LD_PRELOAD cannot be preloaded: ignored.
ERROR: ld.so: object 'libdlfaker.so' from LD_PRELOAD cannot be preloaded: ignored.
ERROR: ld.so: object 'libvglfaker.so' from LD_PRELOAD cannot be preloaded: ignored.
Traceback (most recent call last):
  File "/gpfs/cfel/cmi/scratch/user/zakemil/PECD/pecd/propagate.py", line 1687, in <module>
    psi0 = hydrogen.gen_psi0(params) 
  File "/gpfs/cfel/cmi/scratch/user/zakemil/PECD/pecd/propagate.py", line 1086, in gen_psi0
    fl = open(params['working_dir']+params['ini_state_file'],'r')
FileNotFoundError: [Errno 2] No such file or directory: '/Users/zakemil/Nextcloud/projects/PECD/tests/molecules/h2o/psi0_h2o_3_8_4_2.5_esp_grid_h2o_uhf_631Gss_8_0.2_com.dat'

real	0m1.930s
user	0m1.946s
sys	0m1.013s
