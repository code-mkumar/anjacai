

query_model = {
    "meta-llama-3.1-8b-instruct@q4_k_m": 4,
    "meta-llama-3.1-8b-instruct@q4_k_m:2": 2,
    "meta-llama-3.1-8b-instruct@q4_k_m:3": 1
}

selected_model,query_model= set_model(query_model)

print(selected_model)
print(query_model)
