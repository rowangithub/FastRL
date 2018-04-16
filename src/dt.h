#include <Python.h>
#include <stdio.h>
#include <string>

// Learning abstractions of a neural model
class DT {
	PyObject* dt_module;
	PyObject* dt_klass;
	PyObject* dt_instance;
	std::string nnfile;

public: 
	DT (std::string nnfile) {
		setenv("PYTHONPATH", ".", 1);

		Py_Initialize();
		//PySys_SetArgv(argc, argv);
    	//pName = PyString_FromString(argv[1]);

		//PyObject* sys = PyImport_ImportModule( "sys" );
		//PyObject* sys_path = PyObject_GetAttrString( sys, "path" );
		//PyObject* folder_path = PyUnicode_FromString( "/Users/hezhu/Documents/Research/games/pole/dtextract/python/dtextract/examples" );
		//PyList_Append( sys_path, folder_path );

    	//PySys_SetPath("./dtextract/python/dtextract/examples");

    	dt_module = PyImport_ImportModule("dtextract.python.dtextract.examples.decisiontree");
    	assert(dt_module != NULL);

    	dt_klass = PyObject_GetAttrString(dt_module, "DecisionTree");
    	assert(dt_klass != NULL);

    	dt_instance = PyInstance_New(dt_klass, NULL, NULL);
    	assert(dt_instance != NULL);

    	this->nnfile = nnfile;
	}

	~DT () {
		Py_DECREF(dt_instance);
		Py_DECREF(dt_klass);
		Py_DECREF(dt_module);

		Py_Finalize();
	}

	int learn (std::string datafile) {
		PyObject* result = PyObject_CallMethod(dt_instance, (char*)"learn", (char*)"(ss)", datafile.c_str(), nnfile.c_str());
    	assert(result != NULL);
    	//int ac = PyInt_AsLong(result);
    	Py_DECREF(result);
    	return 1;
	}

	int predict (double* params, int size) {
		PyObject *nParam = PyTuple_New(size);
		assert(nParam != NULL);

        for (int i = 0; i < size; i++) {
            PyTuple_SetItem(nParam, i, PyFloat_FromDouble(params[i]));
        }

		PyObject* result = PyObject_CallMethod(dt_instance, (char*)"predict", (char*)"(O)", nParam);
    	assert(result != NULL);
    	int ac = PyInt_AsLong(result);
    	
    	Py_DECREF(result);
    	Py_DECREF(nParam);

    	return ac;
	}
};