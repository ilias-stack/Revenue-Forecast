from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from forecaster import Forecaster, LSTMForecaster, XGBForecaster
from dotenv import load_dotenv
import os
import smtplib

# Popping the env variables
os.environ.pop('EMAIL_PASS', None)
os.environ.pop('EMAIL_ADD', None)

# Function responsible for making the prediction
def report(model: Forecaster, n_days: int, dataframe: pd.DataFrame, code6_activation_status: list):
    predictions = model.predict(n_days=n_days, data=dataframe, code6_activation_status=code6_activation_status)
    return predictions.ravel()

# Function to send the emails
def send_emails(credentials: list, receivers: list, message: str, image_data: bytes):
    # Unpack the credentials
    email_acc_address, email_acc_password = credentials

    # Defining the subject of the email
    subject = f"Rapport du {CURRENT_DAY}"
    
    # Create SMTP session
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls() 
        server.login(email_acc_address, email_acc_password)  # Login with credentials
        
        for receiver in receivers:
            # Create the message for each receiver
            msg = MIMEMultipart("related")
            msg['From'] = email_acc_address
            msg['To'] = receiver
            msg['Subject'] = subject

            # Attach the email content (message) in HTML format
            msg.attach(MIMEText(message, 'html'))

            # Attach the image to the email
            mime_image = MIMEImage(image_data)
            mime_image.add_header('Content-ID', '<forecast_graph>')
            msg.attach(mime_image)

            # Send the email
            server.sendmail(email_acc_address, receiver, msg.as_string())

        server.quit()
        print("Emails sent successfully!")

    except Exception as e:
        print(f"Error occurred: {e}")


# Used to define the current day in the eyes of the system. Not needed if in real use-case
CURRENT_DAY = "2023-09-27"


if __name__ == "__main__":

    # Load environment variables from .env file
    load_dotenv()
    email_acc_address = os.getenv("EMAIL_ADD")
    email_acc_password = os.getenv("EMAIL_PASS")

    # Extraction of the email addresses
    emails_df = pd.read_excel("email_targets.xlsx")
    emails = emails_df["Email"].values

    # Extraction of the data of the previous days and grouping the data by day
    revenue_df = pd.read_csv("revenue_by_offer.csv", index_col=0)

    # Convert REPORT_DATE to datetime format
    revenue_df["REPORT_DATE"] = pd.to_datetime(revenue_df["REPORT_DATE"])

    grouped_df = revenue_df.groupby(["REPORT_DATE"]).agg({
        "REVENUE": "sum",
        "IS_CODE6_ENABLED": "first",
    }).reset_index()

    # Extraction of the email template
    email_content = open("email_content.html", "r")
    email_template = email_content.read()
    email_content.close()
    
    # Used to extract the future state of the activation and determine the forecast length in the future
    FORECAST_LEN = 31

    # Calculating the forecast
    month_forecast = report(
        model=XGBForecaster(),
        n_days=FORECAST_LEN,
        dataframe=revenue_df[revenue_df["REPORT_DATE"] < CURRENT_DAY],
        code6_activation_status=grouped_df["IS_CODE6_ENABLED"].loc[grouped_df["REPORT_DATE"] >= CURRENT_DAY].iloc[:FORECAST_LEN].values
    )

    # Generate a graph that displays the forecast next to some old graph
    custom_palette = ["#640D5F", "#D91656", "#EE66A6", "#FFEB55"]
    sns.set(style="whitegrid")

    past_data = grouped_df[grouped_df["REPORT_DATE"] <= CURRENT_DAY].tail(FORECAST_LEN)

    plt.plot(past_data["REPORT_DATE"], past_data["REVENUE"], label='Revenue', color=custom_palette[0])

    forecast_dates = pd.date_range(start=CURRENT_DAY, periods=FORECAST_LEN, freq='D')
    plt.plot(forecast_dates, month_forecast, label='Forecast', linestyle='--', color=custom_palette[1])

    plt.axvline(x=pd.to_datetime(CURRENT_DAY), color=custom_palette[2], linestyle='-', label='Prediction Separator')

    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.title("Revenue Prediction vs Historical Data", color=custom_palette[3])

    plt.legend(loc="upper left", frameon=True, shadow=True, facecolor='white')
    plt.xticks(rotation=45)  
    plt.tight_layout()


    # Save the plot to a BytesIO object instead of displaying it
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png') 
    buffer.seek(0)

    # Convert the image to Base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Preparing the email to be sent
    email_content_tosend = email_template.format(
        prediction_day_1=f"{month_forecast[0] / 1e6:.2f}",
        prediction_day_2=f"{month_forecast[1] / 1e6:.2f}",
        weekly_revenue=f"{month_forecast[0:7].sum() / 1e6:.2f}",
        percentage_change=f"{(month_forecast[0] / past_data['REVENUE'].values[-1] - 1) * 100:.2f}"
    )
    
    # Call the send_emails function and passing image binary data
    send_emails([email_acc_address, email_acc_password], emails, email_content_tosend, buffer.getvalue())

    buffer.close() # Closing the buffer after reading from it