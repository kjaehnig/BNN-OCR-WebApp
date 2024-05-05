import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

def neg_loglike(ytrue, ypred):
    return -ypred.log_prob(ytrue)

def divergence(q,p,_):
    return tfd.kl_divergence(q,p)/60000.


def grab_digits_from_canvas(image):
    if image is None:
        print("Error: Image not found. Check the file path.")
        exit()
    if image.shape[2] == 4:
        image = image[:, :, :3]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)

    # Apply GaussianBlur
    blur = cv2.GaussianBlur(gray, (5, 5), 0, 0)
    # print(help(cv2.GaussianBlur))

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(blur, 255., cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow('Threshold', thresh)
    # cv2.waitKey(0)

    # Prepare to crop the contours
    digit_images = []

    # Sort contours by their left most x-coordinate, left to right
    contours = sorted(
        contours,
        key=lambda ctr: (cv2.boundingRect(ctr)[0]))

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        print(x, y, w, h)

        # Make sure the contour area is a likely digit (optional, adjust sizes as needed)
        if w > 10 and h > 25:
            digit = gray[y:y + h, x:x + w]
            # digit_images.append(digit)

        if h > w:  # More height than width, pad width
            pad_size = abs((h - w)) // 2
            digit = cv2.copyMakeBorder(digit, pad_size // 2, pad_size // 2, pad_size, pad_size, cv2.BORDER_CONSTANT,
                                       value=[0, 0, 0])
        else:  # More width than height, pad height
            pad_size = abs(w - h) // 2
            digit = cv2.copyMakeBorder(digit, pad_size, pad_size, pad_size // 2, pad_size // 2, cv2.BORDER_CONSTANT,
                                       value=[0, 0, 0])
        # Resize to 28x28
        resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
        digit_images.append(resized)
        # Draw rectangle around each digit on the original image
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Resize and Normalize Digits
    mnist_digits = []
    for digit in digit_images:
        # Resize to 28x28
        resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
        # Normalize pixel values to 0-1
        normalized = resized / 255.0
        mnist_digits.append(normalized)

    return mnist_digits


def process_image(image_data):
    # Preprocess the canvas image for prediction
    if image_data.shape[2] == 4:
        image_data = image_data[:, :, :3]
    gryimg = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    invimg = gryimg
    img = cv2.resize(invimg, (28, 28),
               interpolation=cv2.INTER_AREA)
    # img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    # img = np.expand_dims(img, axis=-1) / 255.0
    img = img[np.newaxis, :]
    return img


def plot_prediction_probs(probs):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.bar(range(10), probs.squeeze(), tick_label=range(10))
    ax.set_title("BNN Predictions")
    plt.xlabel('Probability')
    plt.ylabel('Digit')
    return fig


def plot_preprocessed_image(img):
    fig, imgax = plt.subplots(figsize=(1.,1.))
    imgax.imshow(img.reshape(28,28, 1), cmap='gray')
    imgax.set_title('What model sees', fontsize=4)
    imgax.tick_params(left=False,
                      bottom=False,
                      labelleft=False,
                      labelbottom=False)
    return fig
# Load the saved Bayesian model


@st.cache_resource
def load_model_into_streamlit():
    loaded_model = load_model('mnist_bnn/',
                   compile=False,)
                   # custom_objects={'neg_loglike':neg_loglike,
                   #                 'divergence':divergence})
    loaded_model.trainable = False
    return loaded_model 

model = load_model_into_streamlit()

# Initialize session state variables if they don't already exist
if 'correct_predictions' not in st.session_state:
    st.session_state.correct_predictions = 0
if 'incorrect_predictions' not in st.session_state:
    st.session_state.incorrect_predictions = 0

# st.write(f"Correct Predictions: {st.session_state.correct_predictions}")
# st.write(f"Incorrect Predictions: {st.session_state.incorrect_predictions}")


if "yes_checkbox_val" not in st.session_state:
    st.session_state["yes_checkbox_val"] = False
if 'no_checkbox_val' not in st.session_state:
    st.session_state['no_checkbox_val'] = False

st.title('Bayesian MNIST Multi-Digit Classifier')

with st.expander("Description"):
    st.write("""
    Bayesian neural networks (BNN) don't fit single value weights when they train. 
    BNNs instead fit distributions with parameters to better describe uncertainty 
    in the data, as well as in the model itself. Prediction probabilities across all of the 
    class labels in a prediction provide a better picture of how sure (or unsure) the 
    model is about it's final classification. This app here uses OpenCV to separate out 
    numbers to perform individual predictions, then reconstructing the resulting number.
    """)

def predict_digit_from_canvas(canvas_data, num_samples):
    if canvas_data is not None:
        # Preprocessing
        img = grab_digits_from_canvas(canvas_data)
        # st.write([ii.shape for ii in img])
        # Prediction
        # pred = model.predict(img, batch_size=num_samples)  # Assume model.predict handles BNN sampling
        # st.write(pred)
        # pred = np.percentile(pred, 50, axis=0)  # Median over samples

        # if len(img)==1:
        #     if show_model_imgs:
        #         st.write("### **What the model sees**")
        #         st.image(img[0].reshape(28,28,1),
        #             clamp=True,
        #             use_column_width='always'
        #                  )
        #
        #     pred = np.array([model(img[0].reshape(1,28,28,1)).numpy().squeeze() for ii in range(num_samples)])
        #     st.write(pred.shape)
        #     st.write(pred)
        #     pred = np.sum(pred, axis=0) / num_samples
        #     # pred = np.exp(pred) / np.sum(np.exp(pred))
        #     pred_digit = [np.argmax(pred)]
        #     return img, pred, pred_digit

        # if len(img) > 1:
        if show_model_imgs:
            allcols = st.columns(len(img))
            for ii, col in enumerate(allcols):
                col.image(img[ii].reshape(28,28,1),
                        clamp=True,
                        use_column_width='always')
        pred = np.zeros((len(img), 10, num_samples))
        for ii in range(num_samples):
            pred[:, :, ii] = model(np.array(img).reshape(-1, 28, 28, 1).astype("float32")).numpy().squeeze()

        # pred = np.array([model(np.array(img).reshape(len(img),28,28,1)).numpy().squeeze() for ii in range(num_samples)])
        st.write(pred.shape)
        # st.write(np.unique(pred))
        pred = np.sum(pred, axis=-1) / num_samples
        st.write(pred.shape)
        # st.write(np.unique(pred))
        pred_digit = ''.join([np.argmax(pred[ii, :]).astype("str") for ii in range(len(img))])
        return img, pred, pred_digit
    return "No digit drawn or image not processed correctly."


def clear_selection():
    for key in st.session_state.keys():
        if key.startswith("User_input_on_prediction"):
            st.session_state[key] = "False"

# col1, col2 = st.columns(2)
# with col1:
with st.container():

    st.write("**Try it out! Draw digits (0-9) on the canvas**")
    # Streamlit canvas for drawing digits
    canvas_result = st_canvas(
        stroke_width=12,
        stroke_color='#ffffff',
        background_color='#000000', 
        height=315, 
        width=700,
        drawing_mode='freedraw', 
        key='canvas',
        update_streamlit=True)


with st.sidebar:
    st.header("Control Panel")
    # Sampling number input
    N = st.slider('N (Number of samples)', min_value=1, max_value=50, value=2)
    if N > 10:
        st.warning("Setting N above 10 may slow down the predictions.")


    plot_all_preds = st.checkbox('Plot digit(s) probabilities?', value=False, key='plot_all_checkbox')

    show_model_imgs = st.checkbox("Show model images?", value=False, key='plot_model_imgs')

pred_digit = None
if pred_digit is None:
    st.session_state.disabled=True

img = None
# Button to submit the drawing for prediction
if st.button('Submit'):
    with st.spinner("**Processing and predicting digit from image...**"):
        img, pred, pred_digit = predict_digit_from_canvas(canvas_result.image_data, N)
        pred_digit_str = ''.join([str(dig) for dig in pred_digit])
        st.write(f"## **Reconstructed number: {pred_digit_str}**")

# plot_model_image = False
# if img is not None and plot_model_image:
#     with st.container():
#         st.write("**What model sees**")
#         st.image(img.reshape(28,28,1), 
#             clamp=True,
#             use_column_width='always')

# plot_all_preds = False
if img is not None and plot_all_preds:
    with st.container():
        if N==1:
            single_sample_warning = "NOTE: Only one sample has been drawn"
            st.write("**Probabilities across possible digits** "+f":red[{single_sample_warning}]")
        else:
            st.write("**Probabilities across possible digits**")
        for ii in range(pred.shape[0]):
            st.write(f"**Probabilities for position {ii}, Classified as a {pred_digit[ii]}**")
            if not isinstance(pred, np.ndarray):
                pred = np.array(pred)
            st.bar_chart(data=pred.squeeze()[ii].T)


def register_prediction_checkbox():
    if st.session_state.yes_checkbox_val:
        st.session_state.correct_predictions += 1
        with st.sidebar:
            st.write("Thanks for responding!")
            st.balloons()
    elif st.session_state.no_checkbox_val:
        st.session_state.incorrect_predictions += 1
        with st.sidebar:
            st.write("Whoops! Let's try again!")

with st.sidebar:
    st.header("Is the model correct?")
    feedback = st.form(
        "Is the model correct?", 
        clear_on_submit=True,
        )

    feedback.checkbox('Yes', value=False, key='yes_checkbox_val')
    feedback.checkbox('No', value=False, key='no_checkbox_val')

    feedback.form_submit_button("Submit", 
        on_click=register_prediction_checkbox,
        disabled=True if img is None else False)


    st.write(f"Correct Predictions: {st.session_state.correct_predictions}")
    st.write(f"Incorrect Predictions: {st.session_state.incorrect_predictions}")

