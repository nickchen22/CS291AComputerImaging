from reconstruction import Reconstruction

pixel_height = 192
pixel_width = 256

if __name__ == '__main__':
	rec = Reconstruction(image_width=pixel_width, image_height=pixel_height, train_directory='test_imgs')
	print("End test")