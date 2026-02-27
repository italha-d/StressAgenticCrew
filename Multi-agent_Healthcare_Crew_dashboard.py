# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 18:54:47 2025

@author: Talha
"""

import os
import io
import base64
import pandas as pd
import datetime
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from dash import dcc, html, Dash, no_update
from dash.dependencies import Input, Output, State

# Machine Learning and Plotting Imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Report generation imports
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# LLM client (assumed available)
import ollama

# ================================
# Agent 2: LLM-Based Clinical Report Generator
# ================================
import os
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

class ClinicalSummaryAgent:
    def __init__(self):
        self.client = ollama.Client()
        self.model = "mistral"
        #self.model = "meditron"
        self.max_tokens = 1000

    def generate_llm_report(self, test_results):
        prompt = f"""
The following clinical analysis has been performed for a participant based on physiological data:

Descriptive Analysis:
{test_results.get('descriptive_analysis')}

Classification Report:
{test_results.get('classification_report')}

Abnormal Patterns:
{test_results.get('abnormal_patterns')}

Based on the above, please provide a comprehensive clinical report in the following structured format:

Clinical Report:
<Provide a comprehensive clinical discussion summarizing the analysis.>

Classification Results:
<Convert the classification report into a paragraph explaining the results.>

Abnormal Patterns:
<Discuss any abnormal patterns detected in the signal. Also, highlight the medical dieseases that are corelated to the prolong stress periods.>

Recommendations:
<Provide a list of actionable recommendations to overcome prolonged stress periods, each on a new line.>
        """
        response = self.client.generate(model=self.model, prompt=prompt)
        return response.response.strip()

    def parse_llm_report(self, llm_report):
        """
        Parse the LLM report into sections based on headings.
        Expected headings are:
          - Clinical Report:
          - Classification Results:
          - Abnormal Patterns:
          - Recommendations:
        """
        sections = {
            "Clinical Report": "",
            "Classification Results": "",
            "Abnormal Patterns": "",
            "Recommendations": ""
        }
        current_section = None
        for line in llm_report.splitlines():
            stripped = line.strip()
            # Check if line is a section header (ending with ':' and matching one of the keys)
            if stripped.endswith(":"):
                header = stripped[:-1]
                if header in sections:
                    current_section = header
                    continue
            if current_section:
                sections[current_section] += line + "\n"
        # Clean up extra whitespace from each section.
        for key in sections:
            sections[key] = sections[key].strip()
        return sections

    def generate_pdf_report(self, test_results, llm_report, participant):
        # Parse the LLM report into structured sections.
        sections = self.parse_llm_report(llm_report)

        pdf_filename = f"LLM_Report_Participant_{participant}.pdf"
        pdf_file = os.path.join(os.getcwd(), pdf_filename)

        doc = SimpleDocTemplate(
            pdf_file,
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )

        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading_style = styles['Heading2']
        normal_style = styles['BodyText']

        story = []

        # Document Header
        story.append(Paragraph(f"Clinical Report for Participant {participant}", title_style))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Generated on: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), normal_style))
        story.append(Spacer(1, 24))

        # Section: Clinical Report
        story.append(Paragraph("Clinical Report", heading_style))
        story.append(Spacer(1, 12))
        clinical_report = sections.get("Clinical Report", "No clinical report available.")
        for para in clinical_report.split("\n"):
            story.append(Paragraph(para.strip(), normal_style))
            story.append(Spacer(1, 12))
        story.append(Spacer(1, 24))

        # Section: Classification Results
        story.append(Paragraph("Classification Results", heading_style))
        story.append(Spacer(1, 12))
        classification_results = sections.get("Classification Results", 
                                                test_results.get("classification_report", "No classification results available."))
        for para in classification_results.split("\n"):
            story.append(Paragraph(para.strip(), normal_style))
            story.append(Spacer(1, 12))
        story.append(Spacer(1, 24))

        # Section: Abnormal Patterns
        story.append(Paragraph("Abnormal Patterns", heading_style))
        story.append(Spacer(1, 12))
        abnormal_patterns = sections.get("Abnormal Patterns", 
                                           test_results.get("abnormal_patterns", "No abnormal patterns available."))
        for para in abnormal_patterns.split("\n"):
            story.append(Paragraph(para.strip(), normal_style))
            story.append(Spacer(1, 12))
        story.append(Spacer(1, 24))

        # Section: Recommendations (each recommendation enumerated on a new line)
        story.append(Paragraph("Recommendations", heading_style))
        story.append(Spacer(1, 12))
        recommendations_text = sections.get("Recommendations", "")
        if recommendations_text:
            # Split recommendations by lines.
            recommendations = []
            for line in recommendations_text.splitlines():
                stripped_line = line.strip()
                if stripped_line:
                    recommendations.append(stripped_line)
            # If not already numbered, enumerate each recommendation.
            if not all(item.lstrip()[0].isdigit() for item in recommendations if item):
                recommendations = [f"{i+1}. {rec}" for i, rec in enumerate(recommendations)]
            for rec in recommendations:
                story.append(Paragraph(rec, normal_style))
                story.append(Spacer(1, 6))
        else:
            story.append(Paragraph("No recommendations available.", normal_style))
        story.append(Spacer(1, 24))

        # Optionally include any plot images, if available.
        image_paths = test_results.get("plot_images", [])
        for img_path in image_paths:
            try:
                img = Image(img_path, width=300, height=300)
                story.append(img)
                story.append(Spacer(1, 12))
            except Exception:
                continue

        doc.build(story)
        return pdf_file

# ================================
# Agent 1: Model Training and Testing
# ================================
class ModelAgent:
    def __init__(self):
        self.model = None
        self.X_test = None
        self.y_test = None
        self.test_data = None  # DataFrame including test samples with IDs

    def train_model(self, df):
        # Preprocess dataset. Expecting:
        #  - Target column 'Label_B'
        #  - Participant identifier in column 'ID'
        #  - Time-series information in columns such as 'time', 'HR', 'respr'
        status_msgs = []
        status_msgs.append("Preprocessing data...")
        df = df.dropna()
        if 'Labels' in df.columns:
            df = df.drop('Labels', axis=1)
        X = df.drop('Label_B', axis=1)
        y = df['Label_B']
        status_msgs.append("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        status_msgs.append("Training RandomForestClassifier model...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        status_msgs.append("Model training complete.")
        self.model = clf
        self.X_test = X_test
        self.y_test = y_test
        self.test_data = X_test.copy()
        self.test_data['Actual_Label'] = y_test.values
        self.test_data['Predicted_Label'] = clf.predict(X_test)

        accuracy = accuracy_score(y_test, self.test_data['Predicted_Label'])
        report = classification_report(y_test, self.test_data['Predicted_Label'])
        # Return status messages along with training metrics
        return {
            "accuracy": accuracy, 
            "classification_report": report, 
            "status": "\n".join(status_msgs),
            "train_test_data": (X_train, y_train, X_test, y_test)
        }

    def test_participant(self, participant_id):
        # Filter the test data for the given participant ID.
        status_msgs = []
        status_msgs.append(f"Filtering test data for participant {participant_id}...")
        if self.test_data is None:
            status_msgs.append("No test data available.")
            return {"status": "\n".join(status_msgs)}
        participant_data = self.test_data[self.test_data['ID'] == participant_id].sort_values('time')
        if participant_data.empty:
            status_msgs.append("No data found for this participant.")
            return {"status": "\n".join(status_msgs)}

        total = len(participant_data)
        stressed = participant_data['Actual_Label'].sum()
        descriptive = f"Total measurements: {total}. Stressed measurements: {stressed}. Non-stressed: {total - stressed}."
        status_msgs.append("Generating classification report for the participant...")
        report = classification_report(participant_data['Actual_Label'], participant_data['Predicted_Label'])
        cm = confusion_matrix(participant_data['Actual_Label'], participant_data['Predicted_Label'])

        # Generate Confusion Matrix Plot
        plt.figure(figsize=(12, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        plt.title(f"Confusion Matrix for Participant {participant_id}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_filename = f"cm_participant_{participant_id}.png"
        plt.savefig(cm_filename,dpi=800)
        plt.close()

                 
        # Heart Rate vs Time Plot
        plt.figure(figsize=(12, 12))
        
        # Split data
        pred_0 = participant_data[participant_data['Predicted_Label'] == 0]
        pred_1 = participant_data[participant_data['Predicted_Label'] == 1]
        
        actual_0 = participant_data[participant_data['Actual_Label'] == 0]
        actual_1 = participant_data[participant_data['Actual_Label'] == 1]
        
        # Predicted (filled circles)
        plt.scatter(pred_0['time'], pred_0['HR'], color='blue', marker='o', label='Predicted 0', alpha=0.7)
        plt.scatter(pred_1['time'], pred_1['HR'], color='red', marker='o', label='Predicted 1', alpha=0.7)
        
        # Actual (triangles)
        plt.scatter(actual_0['time'], actual_0['HR'], color='blue', marker='^', label='Actual 0', edgecolors='k')
        plt.scatter(actual_1['time'], actual_1['HR'], color='red', marker='^', label='Actual 1', edgecolors='k')
        
        plt.title(f"Participant {participant_id} - Heart Rate vs. Time")
        plt.xlabel("Time")
        plt.ylabel("Heart Rate")
        plt.legend()
        
        hr_filename = f"hr_participant_{participant_id}.png"
        plt.savefig(hr_filename, dpi=800)
        plt.close()

        # Respiratory Rate vs Time Plot
        plt.figure(figsize=(12, 12))
        
        pred_0 = participant_data[participant_data['Predicted_Label'] == 0]
        pred_1 = participant_data[participant_data['Predicted_Label'] == 1]
        
        actual_0 = participant_data[participant_data['Actual_Label'] == 0]
        actual_1 = participant_data[participant_data['Actual_Label'] == 1]
        
        plt.scatter(pred_0['time'], pred_0['respr'], color='blue', marker='o', label='Predicted 0', alpha=0.7)
        plt.scatter(pred_1['time'], pred_1['respr'], color='red', marker='o', label='Predicted 1', alpha=0.7)
        
        plt.scatter(actual_0['time'], actual_0['respr'], color='blue', marker='^', label='Actual 0', edgecolors='k')
        plt.scatter(actual_1['time'], actual_1['respr'], color='red', marker='^', label='Actual 1', edgecolors='k')
        
        plt.title(f"Participant {participant_id} - Respiratory Rate vs. Time")
        plt.xlabel("Time")
        plt.ylabel("Respiratory Rate")
        plt.legend()
        
        rr_filename = f"rr_participant_{participant_id}.png"
        plt.savefig(rr_filename, dpi=800)
        plt.close()

        # Assume participant_data is your DataFrame and n is the total number of rows.
        n = len(participant_data)
        
        # Divide the input signal into 3 sections and select the first section.
        if n < 3:
            first_section = participant_data  # Not enough data to split; use all available data.
        else:
            section_size = n // 3  # integer division to get section length.
            first_section = participant_data.iloc[:section_size]
        
        # Calculate the historical mean from the first section.
        #historical_mean_first = first_section['Actual_Label'].mean() * 100
        historical_mean_first = first_section['Predicted_Label'].mean() * 100
        
        # Define a buffer and compute the dynamic threshold.
        buffer = 0
        dynamic_threshold = historical_mean_first + buffer
        
        # Calculate the current stress percentage from the entire dataset.
        #stress_percentage = (participant_data['Actual_Label'].sum() / n) * 100
        stress_percentage = (participant_data['Predicted_Label'].sum() / n) * 100
        
        print(f"Stress Percentage: {stress_percentage:.2f}%")
        print(f"dynamic_threshold: {dynamic_threshold:.2f}%")
        
        # Automatically trigger ClinicalSummaryAgent if current stress exceeds the dynamic threshold.
        if stress_percentage > dynamic_threshold:
            print("Stress detected. Generating report...")
            clinical_agent = ClinicalSummaryAgent()
            llm_text = clinical_agent.generate_llm_report({
                'descriptive_analysis': descriptive,
                'classification_report': report,
                'abnormal_patterns': "Abnormal stress pattern detected."
            })
            pdf_file = clinical_agent.generate_pdf_report({
                'descriptive_analysis': descriptive,
                'classification_report': report,
                'abnormal_patterns': "Abnormal stress pattern detected.",
                'plot_images': [cm_filename, hr_filename, rr_filename]
            }, llm_text, participant_id)
            status_msgs.append("Automatically generated LLM report and saved PDF due to abnormal stress detection.")
        else:
            llm_text = "No LLM report generated."
            status_msgs.append("Stress levels within acceptable range. No report generated.")
        
        status_msgs.append("Testing complete.")
        
        return {
            "descriptive_analysis": descriptive,
            "classification_report": report,
            "abnormal_patterns": "Abnormal stress pattern detected." if stress_percentage > dynamic_threshold else "No abnormal stress pattern detected.",
            "llm_report": llm_text,
            "plot_images": [cm_filename, hr_filename, rr_filename],
            "status": "\n".join(status_msgs)
        }


# ================================
# Dash Dashboard Layout and Callbacks
# ================================
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Stress Analysis Dashboard"),
    dcc.Upload(
        id='upload-dataset',
        children=html.Div(['Drag and Drop or ', html.A('Select a CSV File')]),
        style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
               'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
               'textAlign': 'center', 'margin': '10px'},
        multiple=False
    ),
    html.Button("Train Model", id="train-model-button", n_clicks=0),
    html.Div(id="training-output", style={'margin': '10px', 'color': 'green', 'whiteSpace': 'pre-wrap'}),
    html.Hr(),
    html.H2("Test Model for a Participant"),
    dcc.Dropdown(id="participant-dropdown", placeholder="Select Participant"),
    html.Button("Test Model", id="test-model-button", n_clicks=0),
    html.Div(id="testing-output", style={'margin': '10px', 'whiteSpace': 'pre-wrap'}),
    html.Hr(),
    html.Button("Mannually Generate LLM Report", id="generate-report-button", n_clicks=0),
    html.Div(id="report-output", style={'margin': '10px', 'color': 'blue', 'whiteSpace': 'pre-wrap'}),
    # Stores for passing data between callbacks
    dcc.Store(id='dataset-store'),
    dcc.Store(id='model-agent-store'),
    dcc.Store(id='test-results-store')
])

# Store uploaded dataset
@app.callback(
    Output('dataset-store', 'data'),
    Input('upload-dataset', 'contents'),
    State('upload-dataset', 'filename')
)
def store_dataset(contents, filename):
    if contents is None:
        return no_update
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        return no_update
    return df.to_dict('records')

# Train model callback (Agent 1)
@app.callback(
    [Output("training-output", "children"),
     Output("model-agent-store", "data"),
     Output("participant-dropdown", "options")],
    Input("train-model-button", "n_clicks"),
    State("dataset-store", "data")
)
def train_model(n_clicks, data):
    if n_clicks < 1 or data is None:
        return no_update, no_update, no_update
    status_verbose = "Initiating training process...\n"
    df = pd.DataFrame(data)
    agent = ModelAgent()
    training_results = agent.train_model(df)
    status_verbose += training_results["status"] + "\n"
    accuracy = training_results["accuracy"]
    report = training_results["classification_report"]
    # Save test_data in the agent store (as a dict)
    agent_data = {
        "accuracy": accuracy,
        "report": report,
        "test_data": pd.DataFrame(agent.test_data).to_dict('records')
    }
    # Populate participant dropdown options using the test_data 'ID' column
    test_df = pd.DataFrame(agent.test_data)
    participant_options = [{'label': str(pid), 'value': pid} for pid in test_df['ID'].unique()]
    status_verbose += f"Training complete. Test Accuracy: {accuracy:.2f}\n"
    return status_verbose, agent_data, participant_options

# Test model on a selected participant (Agent 1)
@app.callback(
    [Output("testing-output", "children"),
     Output("test-results-store", "data")],
    Input("test-model-button", "n_clicks"),
    State("participant-dropdown", "value"),
    State("model-agent-store", "data")
)
def test_model(n_clicks, participant, agent_data):
    if n_clicks < 1 or participant is None or agent_data is None:
        return no_update, no_update
    status_verbose = f"Starting testing for participant {participant}...\n"
    test_df = pd.DataFrame(agent_data["test_data"])
    agent = ModelAgent()
    agent.test_data = test_df
    results = agent.test_participant(participant)
    if results is None:
        return f"No test data available for participant {participant}.", no_update
    status_verbose += results.get("status", "") + "\nTesting complete.\n"
    # Prepare summary text
    summary_text = (
        f"Descriptive Analysis:\n{results['descriptive_analysis']}\n\n"
        f"Classification Report:\n{results['classification_report']}\n\n"
        f"Abnormal Patterns:\n{results['abnormal_patterns']}"
    )
    # Convert generated plots to images to display inline
    images_html = []
    for img_path in results['plot_images']:
        if os.path.exists(img_path):
            encoded = base64.b64encode(open(img_path, 'rb').read()).decode('ascii')
            images_html.append(html.Img(src=f"data:image/png;base64,{encoded}",
                                        style={'height': '200px', 'margin': '10px'}))
    return html.Div([html.Pre(status_verbose), html.Pre(summary_text)] + images_html), results

# Generate LLM report callback (Agent 2)
@app.callback(
    Output("report-output", "children"),
    Input("generate-report-button", "n_clicks"),
    State("test-results-store", "data"),
    State("participant-dropdown", "value")
)
def generate_report(n_clicks, test_results, participant):
    if n_clicks < 1 or test_results is None or participant is None:
        return no_update
    status_verbose = "Generating LLM report...\nCalling LLM API...\n"
    agent2 = ClinicalSummaryAgent()
    llm_text = agent2.generate_llm_report(test_results)
    status_verbose += "LLM API call completed.\nSaving PDF report...\n"
    pdf_file = agent2.generate_pdf_report(test_results, llm_text, participant)
    status_verbose += "PDF report saved successfully.\n"
    pdf_link = f"/{os.path.basename(pdf_file)}"
    return html.Div([
        html.Pre(status_verbose),
        html.P("LLM Report Generated."),
        html.A("Download Report", href=pdf_link, target="_blank")
    ])

import webbrowser

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8050
    url = f"http://{host}:{port}"
    
    print(f"\n✅ Dashboard running at: {url}\n")
    webbrowser.open(url)
    
    app.run_server(debug=True, host=host, port=port)