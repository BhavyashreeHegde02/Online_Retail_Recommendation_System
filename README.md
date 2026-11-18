🛍️ Online Retail Recommendation System
A Python-based e-commerce analytics and product recommendation project
built with Streamlit. It analyzes online retail transactions,
explores purchasing patterns, and recommends similar products using
item-based collaborative filtering and cosine similarity.

The project combines data preprocessing, exploratory analysis, model
evaluation support, recommendation logic, and an interactive dashboard
to support product discovery and data-driven retail decisions.
📌 Project Overview
The system uses customer and invoice-level purchase history to identify
products that are commonly purchased by the same customers. A user can
select a product and generate a ranked list of related products.

The main dashboard also presents retail KPIs and sales insights,
including revenue trends, country-level revenue, and top products by
revenue.

🎯 Objectives
- Clean and validate online retail transaction data.
- Explore customer purchasing behaviour and product sales patterns.
- Build an item-based product recommendation approach.
- Calculate product-to-product similarity using cosine similarity.
- Display recommendations with similarity scores and co-purchase
  customer counts.
- Present key retail metrics and visual insights in an interactive
  dashboard.

🧰 Tech Stack
- Language: Python
- Application: Streamlit
- Data processing: Pandas, NumPy
- Recommendation and similarity: SciPy sparse matrices,
  scikit-learn
- Visualisation: Plotly, Matplotlib
- Excel support: OpenPyXL

🧠 Recommendation Approach
The primary dashboard (app.py) uses item-based collaborative
filtering.
1. Retail transactions are validated and prepared.
2. An item--customer interaction matrix is created.
3. By default, interactions are binary (whether a customer purchased an
   item); quantity-weighted interactions can also be selected.
4. Product vectors are normalized and compared using cosine similarity.
5. The system retrieves the Top-N similar products for the selected
   product.
6. Results include:
   - Similarity: cosine similarity between product vectors.
   - CoPurchaseCustomers: number of customers who purchased both
     products.
The dashboard also allows the user to configure minimum customer support
per product and the number of recommendations to return.
The repository also contains app2.py, an earlier/alternative
Streamlit implementation. It builds an invoice-by-product basket
matrix and applies item-based cosine similarity to recommend the top
five related products.

📊 Dashboard Features
- Retail KPIs: total revenue, unique customers, unique products,
  and unique invoices.
- Revenue trend: revenue over time.
- Country analysis: top countries by revenue.
- Product analysis: top products by revenue.
- Data preview: inspect a sample of the prepared transaction data.
- Product recommendations: select an item and generate Top-N
  related products.
- Explainability: understand similarity scores and co-purchase
  counts.
- Architecture view: visual overview of the data-to-recommendation
  workflow.

🗂️ Repository Structure
  File                                Purpose
  app.py                            Main Streamlit dashboard with KPIs,
analytics, recommendations,
explainability, and architecture
tabs.

  app2.py                           Alternative Streamlit app with
dataset overview, visual analysis,
and product recommendation
sections.

  recommender_backend.py            Data preparation, item--customer
matrix construction, and Top-N
recommendation functions.

  retail_cleaning_pipeline.py       Retail data-cleaning pipeline.

  run_cleaning_pipeline.py          Script to run the cleaning
pipeline.

  1_Graphs_Section.py               Graphs and exploratory analysis
section.

  2_Model_Evaluation_Section.py     Model evaluation section.

  OnlineRetail.xlsx                 Retail transaction workbook
included in the repository.
requirements.txt                  Python package dependencies.
⚙️ Getting Started
1. Clone the repository
git clone https://github.com/BhavyashreeHegde02/Online_Retail_Recommendation_System.git
cd Online_Retail_Recommendation_System
2. Create a virtual environment
Windows
python -m venv .venv
.venv\Scripts\activate
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Prepare the cleaned dataset
The main dashboard (app.py) defaults to a file named
OnlineRetail_cleaned.xlsx. Run the cleaning pipeline and ensure the
resulting cleaned workbook is available at that path, or set the
RETAIL_DATA_PATH environment variable to the actual cleaned workbook
location.

The repository also includes OnlineRetail.xlsx, which is used as a
fallback input by app2.py.
5. Launch the main dashboard
streamlit run app.py
To run the alternative implementation:
streamlit run app2.py

📁 Dataset Fields
The application works with retail transaction fields such as:
- InvoiceNo -- transaction/invoice identifier.
- StockCode -- product identifier.
- Description -- product name or description.
- Quantity -- number of items purchased.
- InvoiceDate -- transaction date and time.
- UnitPrice -- price per item.
- CustomerID -- customer identifier.
- Country -- country associated with the transaction.
The main dashboard also derives or uses TotalPrice for revenue
analysis.

🔍 Data Preparation
The recommender backend prepares the input data by:
- Parsing invoice dates and numeric transaction fields.
- Deriving TotalPrice from quantity and unit price when it is not
  already available.
- Removing records without a valid customer or invoice.
- Keeping positive quantities and positive unit prices.
- Preparing product descriptions for the recommendation matrix.
- Filtering products by minimum customer support before recommendation
  generation.

📈 Evaluation and Explainability
The main dashboard stores recommendation-related labels and scores for
its evaluation workflow. The repository includes a separate model
evaluation section.

Specific evaluation results and metric values are not stated here
because they should be taken directly from the executed evaluation
output rather than assumed.

🚀 Possible Future Improvements
- Add screenshots or a short demo of the dashboard.
- Document evaluation metrics and measured results.
- Add automated tests for data preparation and recommendation
  functions.
- Provide a reproducible sample cleaned dataset or a clear
  data-generation step.
- Add deployment instructions for hosting the Streamlit application.

👩‍💻 Author
Bhavyashree Hegde
MCA Student --- PES University, Bengaluru
- GitHub: BhavyashreeHegde02
- LinkedIn: Connect with
  me
- Email: bhavyashreehegde0@gmail.com

⭐ If you find this project interesting, feel free to explore the
repository and share feedback.
