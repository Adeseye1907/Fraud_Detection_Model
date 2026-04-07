FileNotFoundError: [Errno 2] No such file or directory: 'fraud_detection_model.joblib'

File "C:\Users\user\Desktop\Machine Learning\Project\Fruad detection\fraud.py", line 49, in <module>
    model = load_model()
File "C:\Users\user\Desktop\Machine Learning\streamlit3_13\Lib\site-packages\streamlit\runtime\caching\cache_utils.py", line 227, in __call__
    return self._get_or_create_cached_value(args, kwargs, spinner_message)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\user\Desktop\Machine Learning\streamlit3_13\Lib\site-packages\streamlit\runtime\caching\cache_utils.py", line 269, in _get_or_create_cached_value
    return self._handle_cache_miss(cache, value_key, func_args, func_kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\user\Desktop\Machine Learning\streamlit3_13\Lib\site-packages\streamlit\runtime\caching\cache_utils.py", line 328, in _handle_cache_miss
    computed_value = self._info.func(*func_args, **func_kwargs)
File "C:\Users\user\Desktop\Machine Learning\Project\Fruad detection\fraud.py", line 47, in load_model
    return joblib.load(MODEL_FILE)
           ~~~~~~~~~~~^^^^^^^^^^^^
File "C:\Users\user\Desktop\Machine Learning\streamlit3_13\Lib\site-packages\joblib\numpy_pickle.py", line 735, in load
    with open(filename, "rb") as f:
         ~~~~^^^^^^^^^^^^^^^^

# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":
    main()