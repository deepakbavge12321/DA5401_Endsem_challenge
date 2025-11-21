# DA5401 – End Semester Data Challenge: Metric Learning for Conversational AI Evaluation  

**Name:** Bavge Deepak Rajkumar

**Roll Number:** NA22B031 

---

## Overview  

This project focuses on building a **metric learning–style regression model** to predict how well a prompt–response pair aligns with a given evaluation metric.  
Each training sample contains:
- a **metric name**,  
- a **user prompt**,  
- a **model response**,  
- an optional **system prompt**,  
- and a **fitness score** in the range **0–10**.

The challenge lies in combining:
1. a **fixed 768-dimensional embedding** representing the metric definition, and  
2. the **textual content** of the conversation  

to accurately predict the relevance score.  
The task essentially tests how well we can fuse embeddings and text features for a supervised regression problem.

---

## Dataset Used  

You must manually download the following files and place them next to `a.ipynb`:

- `train_data.json`  
- `test_data.json`  
- `metric_names.json`  
- `metric_name_embeddings.npy`  

These contain:
- Conversation text for each test case  
- Metric names and their corresponding fixed embeddings  
- Ground-truth fitness scores (only for training)  

There is no direct raw text for metric definitions, only numerical embeddings.

---

## Project Breakdown  

### **1. Data Loading and Preprocessing**

- Loaded all JSON files and converted them into DataFrames.  
- Combined `system_prompt`, `user_prompt`, and `response` into a single text field.  
- Checked for missing values (mainly missing system prompts) and treated missing entries as empty strings.  

### **2. Text Feature Construction**

- Applied **TF–IDF vectorization** using unigrams and bigrams.  
- Reduced dimensionality using **Truncated SVD (512 components)** to obtain dense text embeddings.  

### **3. Metric Embedding Integration**

- Each metric name is mapped to a **768-dimensional embedding**.  
- These embeddings are treated as fixed representations of the metric semantics.  
- Final feature vector = **metric embedding (768) + text embedding (512) = 1280 dimensions**.

---

### **4. Regression Modeling**

The final model used was **XGBoost Regressor**, trained on the 1280-dimensional combined features.

- Train–validation split: **85% / 15%**  
- Objective: `reg:squarederror`  
- Early stopping implemented to prevent overfitting  
- Final model retrained on full training data using the best iteration  

### **5. Evaluation & Output**

- Predictions were rounded and clipped to **0–10** to match label format.  

---

## How to Run  

1. Download the four required dataset files:  
 - `train_data.json`  
 - `test_data.json`  
 - `metric_names.json`  
 - `metric_name_embeddings.npy`  

2. Place them **in the same directory** as `a.ipynb`.
