import streamlit as st
from PIL import Image
import fast_style_model as fsm

# function: to select desired img_shape
def get_img_shape(output_shapes,img_shape):
    if (img_shape == output_shapes[0]):
        return (2160,2160)
    elif (img_shape == output_shapes[1]):
        return (1440,1440)
    elif (img_shape == output_shapes[2]):
        return (1080,1080)
    elif (img_shape == output_shapes[3]):
        return (720,720)
    elif (img_shape == output_shapes[4]):
        return (480,480)
    elif (img_shape == output_shapes[5]):
        return (360,360)
    elif (img_shape == output_shapes[6]):
        return (256,256)

def main():
    st.title("Image Diffusion App")
     # Predefined output image shapes
    output_shapes = ["2160x2160","1440x1440","1080x1080","720x720", "480x480", "360x360","256x256"]  # Add more options as needed

    # Add file upload widgets
    # Dropdown select box for output image shape
    content_img_shape = st.selectbox("Select desired Content image shape:", output_shapes,key = 0)
    content_img_shape = get_img_shape(output_shapes,content_img_shape)
    image1 = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"])
    # Dropdown select box for output image shape
    style_img_shape = st.selectbox("Select desired Style image shape recommended(256x256):", output_shapes,key = 1)
    style_img_shape = get_img_shape(output_shapes,style_img_shape)
    image2 = st.file_uploader("Upload Style Image", type=["jpg", "png", "jpeg"])
    print('style img type:',type(style_img_shape))

    stylized_img_shape = st.selectbox("Select desired Diffused image shape:", output_shapes,key = 2)
    stylized_img_shape = get_img_shape(output_shapes,stylized_img_shape)
  
    

    if image1 and image2:
        # Display the original images
        st.subheader("Original Images")
        st.image(image1, caption="Content Image ", use_column_width=True)
        st.image(image2, caption="Style Image", use_column_width=True)

        # Process the images with your ML model
        process_button = st.button('Process Image')
        if process_button:
            #==============================================================================
            with st.spinner('Processing...'):
                img1 = image1
                img2 = image2
                # Handle image processing here using your ML model
                #print('loading Diffusion Model...')
                dif_model = fsm.get_model('Style_Diffusion_Model/')
                #print('Diffusion Model succesffully loaded')
                print('Preprocessing Content img...')
                content_img = fsm.load_local_img(img1,img_size = content_img_shape)
                #print('Preprocessed Content img successfully')
                #print('Preprocessing Style img...')
                style_img = fsm.load_local_img(img2,img_size = style_img_shape)
                #print('Preprocessed Style img successfully')
                #print('Getting diffused img...')
                stylized_img = fsm.get_stylized_image(dif_model,content_img,style_img)
                #print('Diffused diffused img successfully')
                #print('Saving diffused img...')
                fsm.save_img(stylized_img,image_size=stylized_img_shape)
                #print('Diffused img saved successfully')
                processed_image_path = 'diffused_img.png'
                #==============================================================================
                #diffused_image = your_ml_module.process_images(image1, image2)

                # Display the diffused image
                st.subheader("Diffused Image")
                st.image(processed_image_path, caption="Diffused Image", use_column_width=True)

                  # Add a download button
            download_button = st.download_button(
                label="Download Processed Image",
                data=get_image_bytes(processed_image_path),
                file_name=processed_image_path,
            )

def get_image_bytes(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return image_bytes


if __name__ == "__main__":
    main()
