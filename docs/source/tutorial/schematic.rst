Schematic Generator
===================

In this tutorial, you will create a differential Gm cell schematic generator and a transient testbench generator, then
generate the schematic and testbench and run the simulation in BAG.  In this tutorial, :envvar:`$BAG_FRAMEWORK` refers
to the BAG repository location.  At Berkeley, this directory is :file:`/tools/projects/eeis/BAG_2.0/BAG_framework`.

#. Check out a BAG workspace directory for your process technology.  This tutorial will use TSMC 65 GP as an example.
   If you need a BAG workspace for your technology, contact one of the BAG developers.  To check out the directory
   from git, type the following on the command line:

    .. code-block:: bash

        > git clone git@bwrcrepo.eecs.berkeley.edu:BAG/BAG2_TSMC65_tutorial.git
        > mv BAG2-TSMC65GP-WORKSPACE bag_tutorial
        > cd bag_tutorial

   Note that we renamed the workspace to be :file:`bag_tutorial`.  Feel free to use another name.

#. On the command line, type:

    .. code-block:: bash

        > source .cshrc
        > virtuoso &

   This sets up the environment variables for virtuoso and BAG, then starts a virtuoso instance.

#. To design a circuit, we first need a schematic generator.  A schematic generator is already provided for you.  We'll
   take a look at it in detail so you should be able to create your own.  Open the cellview
   :file:`demo_templates/gm/schematic`:

    .. figure:: ./figures/gm_schematic.png
        :align: center
        :figclass: align-center

        Differential gm cell schematic generator.

   A schematic generator is just like any other normal schematic, except it uses transistors in ``BAG_prim`` library
   instead of foundry transistors.  This is the property window of one of the transistors:

    .. figure:: ./figures/tran_prop.png
        :align: center
        :figclass: align-center

        ``BAG_prim`` transistor properties

   and let's see the ``BAG_prim`` transistor schematic:

    .. figure:: ./figures/tran_schematic.png
        :align: center
        :figclass: align-center

        ``BAG_prim`` transistor schematic

   We see that ``BAG_prim`` transistors are simply wrappers around the foundry transistors.

   For more information about schematic generators, see :doc:`/overview/schematic`.

#. Now that we have a schematic, we also want a testbench to verify the performance of this gm cell.  A very simple
   testbench is provided for you.  Open the cellview :file:`demo_testbenches/gm_tb_tran/schematic`:

    .. figure:: ./figures/gm_testbench.png
        :align: center
        :figclass: align-center

        Differential gm cell testbench.

   This testbench connects the Gm cell to a resistor load, applies a square wave input, and measures the output
   waveform.

   Now, open the property window of the ``XDUT`` instance:

    .. figure:: ./figures/testbench_dut.png
        :align: center
        :figclass: align-center

        Testbench device-under-test (DUT) properties.

   We see that the Gm cell is an instance of the schematic generator.  When a new schematic is generated, BAG will
   create a copy of this testbench and replace the ``XDUT`` instance.

   All BAG testbenches are simulated using ADE-XL.  This is to make process corner and parametric sweeps easier.  If
   you are unfamiliar with ADE-XL, consult the Virtuoso documentation.

   For more information on how to make a testbench for BAG, see :doc:`/overview/testbench`.

#. Now, we'll go over how to write a design script for this Gm cell and use BAG to generate new schematics.

   First, we need to start a BAG server within Virtuoso.  In the CIW window (the window that shows log messages), type
   the following:

    .. code-block:: none

        load("start_bag.il")

   If you see the following 3 lines in the CIW window, then BAG server has started successfully.

    .. figure:: ./figures/bag_server_start.png
        :align: center
        :figclass: align-center

        BAG server start message

   Next, we need to start a simulation server that handles simulation requests from BAG. On command line, type:

    .. code-block:: bash

        > ./sim_server.sh &

   If you see the following window pop up, the simulation server started successfully.  To stop the simulation
   server from running, simply close this window or press the "Exit" button.

    .. figure:: ./figures/bag_sim_server.png
        :align: center
        :figclass: align-center

        Simulation server window
  
   Then, on command line, type:

    .. code-block:: bash

        > ./start_bag.sh

   You should see the IPython interpreter starting up.  In the interpreter, type:

    .. code-block:: none

        In [1]: run -i demo_scripts/bag_import.py

   to executes the script :file:`bag_import.py`.  When it completes without errors, type:

    .. code-block:: none

        In [2]: exit

   to exit IPython.  Let's open :file:`bag_import.py` to see what it did.

   The script :file:`bag_import.py` has only two important lines:

    .. code-block:: python

        prj = bag.BagProject()
        prj.import_design_library(lib_name)

   The first line creates a :class:`~bag.BagProject` instance using the configurations in
   :file:`bag_tutorial/bag_config.yaml` (See :doc:`/setup/setup` to learn more about this file).  The second line
   imports all schematic generators in the library :file:`demo_templates` from Virtuoso and create default Python design
   modules.

#. Open the file :file:`bag_tutorial/BagModules/demo_templates/gm.py` to see the generated design module.  This is
   where you should put your design algorithms.  A filled out design module is already done for you and is located at
   :file:`$BAG_FRAMEWORK/tutorial/demo_templates/gm.py`.  Open this file and compare the two.

   The most important method is :func:`~bag.design.Module.design`.  In this method, you should call the
   :func:`~bag.design.Module.design` method of all the instances in the schematic to set their parameters.  Notice that
   the ``BAG_prim`` transistor design method takes 4 arguments, the width, length, number of fingers, and design intent
   (which translates to transistor threshold flavor).  Other methods are for layout generation, and will not be covered
   in this tutorial.  See :doc:`/overview/design` for more details about design modules.

   Now, copy the complete design module and replace the one generated by :file:`bag_import.py`.

#. With a filled out design module, you can now run the design script.  On the command line, type:

    .. code-block:: bash

        > ./start_bag.sh

   In the IPython console, type:

    .. code-block:: none

        In [1]: run -i bag_dsn.py

   You should see the following output:

    .. code-block:: none

        In [1]: run -i bag_dsn.py
        creating BAG project
        designing module
        design parameters:
        {'input_intent': 'fast',
         'lch': 6e-08,
         'ndum_extra': 1,
         'nf': 4,
         'tail_intent': 'standard',
         'win': 6e-07,
         'wt': 4e-07}
        implementing design with library demo_1
        creating testbench demo_1__gm_tb_tran
        setting testbench parameters
        committing testbench changes
        running simulation
        Starting simulation.  You may press Ctrl-C to abort.
        Simulation took 29.5 s
        Simulation log: /tmp/skillOceanTmpDCOGJZ/ocnLog6NaI9R
        Result directory: /tmp/skillOceanTmpDCOGJZ/bag_sim_datavbqmSl
        loading results
        output waveforms: ['outac']
        outac sweep parameters order: ['corner', 'cload', 'time']
        plotting waveform with parameters:
        corner = ff
        cload = 1e-13

        In [2]:

   And a plot window should show up.  Now, open :file:`bag_dsn.py` to see the magic.  First, focus on these 5 lines:

    .. code-block:: python

        prj = bag.BagProject()
        dsn = prj.create_design_module(lib_name, cell_name)
        dsn.design(**params)
        dsn.update_structure()
        prj.implement_design(impl_lib, dsn)

   The first line creates a :class:`~bag.BagProject` instance just like in :file:`bag_import.py`.  The second line
   creates a design module instance for ``demo_templates/gm``, which we defined earlier in the file
   :file:`bag_tutorial/BagModules/serdes_templates/gm.py`.  The third line calls the design method to create a new
   design, the fourth line calls :func:`~bag.design.Module.update_structure` on the design module instance, which
   prepares it for implementation.  Finally, the fifth line implements the design module by creating a new virtuoso
   schematic.  You can open the cellview ``demo_1/gm/schematic`` in virtuoso to see the implemented schematic.

   The next section creates a new testbench and run the simulation:

    .. code-block:: python

        tb = prj.create_testbench(tb_lib, tb_cell, impl_lib, cell_name, impl_lib)

        tb.set_parameter('tsim', 3e-9)
        tb.set_parameter('rload', 1000)
        tb.set_parameter('tper', 250e-12)
        tb.set_parameter('tr', 15e-12)
        tb.set_parameter('vamp', 120e-3)
        tb.set_parameter('vbias', 0.45)
        tb.set_parameter('vcm', 0.7)
        tb.set_parameter('vdd', 1.0)
        tb.set_parameter('tstep', 1e-12)
        tb.set_sweep_parameter('cload', values=[5e-15, 20e-15, 100e-15])

        tb.set_simulation_environments(['tt', 'ff'])

        tb.add_output(plot_wvfm, """getData("/OUTAC" ?result 'tran)""")

        tb.update_testbench()

        tb.run_simulation()

   The first line copies the testbench we looked at before to library ``impl_lib``, then replaces all device-under-test
   instances with the generated schematic.  The following lines then set the testbench parameter values.  Notice that it
   defines a parametric sweep for ``cload``, and also a process corner sweep.  Next, it tells the testbench to save
   the given Virtuoso calculator expression in ``plot_wvfm``, then calls :func:`~bag.core.Testbench.update_testbench` to
   commit changes to the testbench settings in Virtuoso.  You can open the the cellview ``demo_1/gm_tb_tran/adexl`` to
   see how these changes are written to Virtuoso.  Finally, the last command starts the simulation run.  To learn more
   about configuring testbenches in BAG and running simulations, see :doc:`/overview/testbench`.

   The last section of the script demonstrates how to read simulation results back in Python:

   .. code-block:: python

        results = bag.data.load_sim_results(tb.save_dir)
        vout = results[plot_wvfm]

        par1 = vout.sweep_params[0]
        par2 = vout.sweep_params[1]
        idx1 = -1
        idx2 = -1
        tvec = results['time']
        vvec = vout[idx1,idx2,:]

        plt.figure(1)
        plt.plot(tvec, vvec)
        plt.show(block=False)

   The first line calls :func:`~bag.data.load_sim_results` on the testbench attribute
   :attr:`~bag.core.Testbench.save_dir`, which is the directory containing simulation results.  This method returns a
   Python dictionary from output names/sweep parameter names to their data.  Each output is a numpy array with an extra
   attribute ``sweep_params``, which is a list of the sweep parameters corresponding to each dimension.  The result
   dictionary also have an entry ``sweep_params``, which contains another dictionary that maps output signal names to
   their swee parameters.  The sweep values are stored as 1D numpy arrays in the result dictionary.  Feel free to play
   with the ``results`` variable in the IPython console to explore the data structure.

Congratulations!  You've done your first BAG schematic!  To have a more thorough understanding of BAG and learn more
advanced topics, see :doc:`/overview/overview`.
