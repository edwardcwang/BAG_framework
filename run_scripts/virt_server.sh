#!/bin/tcsh

# hack to get iPDK pycells to work.  Berkeley specific setting.
if ($?PYTHONPATH) then
    setenv PYTHONPATH_PYCELL "${PYTHONPATH}"
endif
setenv PYTHONPATH "${BAG_FRAMEWORK}"

set cmd = "-m bag.virtuoso run_skill_server"
set min_port = 5000
set max_port = 9999
set port_file = "BAG_server_port.txt"
set log = "skill_server.log"

set cmd = "${BAG_PYTHON} ${cmd} ${min_port} ${max_port} ${port_file} ${log}"
exec $cmd
