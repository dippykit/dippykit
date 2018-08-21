Installation Guide
============================================================
This guide assumes that you do not have a subversion of 
Python 3 installed nor do you have PyCharm installed.

This guide does not make use of virtual environments. If you 
are familiar with Python and prefer to install ``dippykit`` via
a virtual environment, you are free to do so.

Setting Up the Environment
------------------------------------------------------------
#. Install any subversion of Python 3. The download for 
   the latest version of Python can be found on `the Python 
   website <https://www.python.org/downloads/>`_. On Windows 
   and Mac, installation should be as simple as executing 
   the downloaded file and completing the installation 
   wizard.

#. We recommend the use of the PyCharm IDE from JetBrains.
   Follow the instructions on `the PyCharm website 
   <https://www.jetbrains.com/pycharm/download/>`_ 
   to download PyCharm (Community version is acceptable).
   On Windows and Mac, installation should be as simple as 
   executing the downloaded file and completing the 
   installation wizard.

Installing ``dippykit``
------------------------------------------------------------
#. We recommend installing ``dippykit`` from PyPI. From the 
   command line, enter the command 
   "``pip install dippykit``". This may take a while, so be 
   prepared to  wait.

Verifying Your Installation
------------------------------------------------------------
#. From the command line, enter the command "``python``". This 
   will convert your terminal into an interactive Python 
   interpreter where you can run Python code line by line.

   - In some cases (e.g. if a version of Python 2 is 
     already installed on your machine), you will need to 
     use the "``python3``" or other command to open an 
     interactive Python *3* interpreter. To exit the 
     interpreter, simply enter the "``exit()``" command. 
     Right before the interpreter opens, it should display 
     the version of Python it is using. Ensure that this 
     version is your subversion of Python 3.

#. Enter the following lines of code into the interpreter:

     >>> import dippykit as dip
     >>> dip.window_2d(8, 'rect', dim=(2, 4))
     array([[0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
            [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
            [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
            [0.   , 0.   , 0.125, 0.125, 0.125, 0.125, 0.   , 0.   ],
            [0.   , 0.   , 0.125, 0.125, 0.125, 0.125, 0.   , 0.   ],
            [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
            [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
            [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ]])
      
#. If the above array was displayed, congratulations - your 
   installation was successful!

Creating a New PyCharm Project
------------------------------------------------------------
#. Open PyCharm.
#. Click on **Create New Project**.
#. After selecting an appropriate location to store the 
   project's files, click on **Project Interpreter**.
#. Select **Existing Interpreter**
#. If the recently installed Python 3 executable is not 
   listed automatically, click the settings/gear button to 
   the right. Select **Add Local** and navigate to wherever
   your executable was installed.
#. Click **Create**.
#. Congratulations - you now have a new PyCharm project. 
   From here, you can take advantage of all the benefits of 
   the IDE.

