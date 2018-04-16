#include "dt.h"
/*#include <iostream>

int main () {
	std::string nnfile = "agent330.network.json";
	DT dt(nnfile);
	dt.learn ("agent330.network.data");
	double params[4];
	params[0] = params[1] = params[2] = params[3] = 100;
	int res = dt.predict(params, 4);
	std::cout << "result = " << res << "\n";
	res = dt.predict(params, 4);
	std::cout << "result = " << res << "\n";
	return 0;
}*/

/*int main()
{
    setenv("PYTHONPATH", ".", 1);

    Py_Initialize();

    PyObject* module = PyImport_ImportModule("multiply");
    assert(module != NULL);

    PyObject* klass = PyObject_GetAttrString(module, "math");
    assert(klass != NULL);

    PyObject* instance = PyInstance_New(klass, NULL, NULL);
    assert(instance != NULL);

    PyObject* result = PyObject_CallMethod(instance, "add", "(ii)", 1, 2);
    assert(result != NULL);

    printf("1 + 2 = %ld\n", PyInt_AsLong(result));

    Py_Finalize();

    return 0;
}*/

/*#include <Python.h>
int main(int argc, char *argv[])
{
    PyObject *pName, *pModule, *pDict, *pFunc;
    PyObject *pArgs, *pValue;
    int i;

    if (argc < 3) {
        fprintf(stderr,"Usage: call pythonfile funcname [args]\n");
        return 1;
    }

    Py_Initialize();
    PySys_SetArgv(argc, argv);
    pName = PyString_FromString(argv[1]);

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, argv[2]);

        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New(argc - 3);
            for (i = 0; i < argc - 3; ++i) {
                pValue = PyInt_FromLong(atoi(argv[i + 3]));
                if (!pValue) {
                    Py_DECREF(pArgs);
                    Py_DECREF(pModule);
                    fprintf(stderr, "Cannot convert argument\n");
                    return 1;
                }
                PyTuple_SetItem(pArgs, i, pValue);
            }
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);
            if (pValue != NULL) {
                printf("Result of call: %ld\n", PyInt_AsLong(pValue));
                Py_DECREF(pValue);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return 1;
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", argv[2]);
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", argv[1]);
        return 1;
    }
    Py_Finalize();
    return 0;
}*/