Collaboration with BAG
======================

In this tutorial, you will use pre-written schematic, layout, and testbench generators to create a new design and a
testbench, then run post-extraction simulation with the generated designs.  After this tutorial, you should have a
good idea of how to use other people's generators.

#. Check out a BAG workspace directory for your process technology.  This tutorial will use TSMC 65 GP as an example.
   If you need a BAG workspace for your technology, contact one of the BAG developers.  To check out the directory
   from git, type the following on the command line:

    .. code-block:: bash

        > git clone git@bwrcrepo.eecs.berkeley.edu:BAG/BAG2_TSMC65_tutorial.git
        > mv BAG2-TSMC65GP-WORKSPACE bag_tutorial
        > cd bag_tutorial
        > source .cshrc

   Note that we renamed the workspace to be :file:`bag_tutorial`.  Feel free to use another name.

#. Next, we need to check out the generators.  For good practice, every designer should keep a repository of the
   generators they're working on, and other will check out the repository to get the generator code.  In :program:`git`,
   this is most naturally done with use of submodules, which lets your repository reference other's repositories.

   When you first check out a repository, you need to explicitly tell git to populate the submodules.  On the command
   line, type:

    .. code-block:: bash

        > git submodule init
        > git submodule update

   This will initialize then check out two repositories, :file:`bag_serdes_burst_mode` and :file:`BAG2_TEMPLATES_EC`.
   :file:`bag_serdes_burst_mode` contains schematic and testbench generators for burst mode serdes links, and
   :file:`BAG2_TEMPLATES_EC` contains layout generators for various analog circuits.

#. Now that we have the generators, we need to modify several configuration files to let BAG know where to find them.
   open the file :file:`cds.lib`, you will see the following two lines:

    .. code-block:: text

        DEFINE serdes_bm_templates $BAG_WORK_DIR/bag_serdes_burst_mode/serdes_bm_templates
        DEFINE serdes_bm_testbenches $BAG_WORK_DIR/bag_serdes_burst_mode/serdes_bm_testbenches

   these two lines tells Virtuoso where to find the schematic and testbench generators.

#. Open the file :file:`bag_libs.def`, you will see the following line:

    .. code-block:: text

        serdes_bm_templates $BAG_WORK_DIR/bag_serdes_burst_mode/BagModules

   this line tells :py:obj:`bag` where it can find the design modules of the schematic generators.

#. Open the file :file:`start_bag.sh`, you will see the following line:

    .. code-block:: bash

        setenv PYTHONPATH "${BAG_WORK_DIR}/BAG2_TEMPLATES_EC:${BAG_TECH_CONFIG_DIR}"

   this line adds :file:`BAG_2_TEMPLATES_EC` and the technology configuration folder to :envvar:`$PYTHONPATH`, so
   :py:obj`bag` can find the layout generators.

#. Now that the configuration files are set up, we are ready to run the code.  Start Virtuoso in the directory, then
   in the CIW window (the window that shows log messages), type the following:

    .. code-block:: none

        load("start_bag.il")

   this starts the BAG server.  Then, on the command line, type:

    .. code-block:: bash

        > ./sim_server.sh &

   to start the simulation server.  Then, o nthe command line, type:

    .. code-block:: bash

        > ./start_bag.sh

   to start the IPython interpreter.  Once you're in the interpreter, type:

    .. code-block:: none

        In [1]: run -i demo_scripts/diffamp_tran.py

   this will create a schematic, layout, and testbench in library ``serdes_bm_1``, run LVS and RCX, run
   post-extraction transient simulation, then import the data back to Python and plot the output waveform.  If you
   see a sinusodial waveform plot, the tutorial has finished successfully.

   to see how each of these steps is done, read the script :file:`demo_scripts/diffamp_tran.py`.
