from utils import read_and_save
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.initializers import RandomNormal
from keras import Input, Model
from keras.layers import Concatenate, Conv2D, BatchNormalization
from keras.layers import LeakyReLU, Activation, Dropout, Conv2DTranspose
from keras.optimizers import Adam
from keras.models import load_model
import cv2 as cv


class Pix2PixModel:
    @staticmethod
    def name():
        return 'Pix2PixModel'

    def __init__(self, rn, image_shape=(512, 1024, 3)):
        self.image_shape = image_shape
        self.d_model = None
        self.g_model = None
        self.gan_model = None
        self.run_number = rn

    # define the discriminator model
    def create_discriminator(self):
        # empty random tensor
        init = RandomNormal(stddev=0.02)
        # tensor for the pose image
        in_src_image = Input(shape=self.image_shape)
        # tensor for the target image, the real frame
        in_target_image = Input(shape=self.image_shape)
        # concatenate images channel-wise
        merged = Concatenate()([in_src_image, in_target_image])
        # C64
        layers = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
        layers = LeakyReLU(alpha=0.2)(layers)
        # C128
        layers = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layers)
        layers = BatchNormalization()(layers)
        layers = LeakyReLU(alpha=0.2)(layers)
        # C256
        layers = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layers)
        layers = BatchNormalization()(layers)
        layers = LeakyReLU(alpha=0.2)(layers)
        # C512
        layers = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layers)
        layers = BatchNormalization()(layers)
        layers = LeakyReLU(alpha=0.2)(layers)
        # second last output layer
        layers = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(layers)
        layers = BatchNormalization()(layers)
        layers = LeakyReLU(alpha=0.2)(layers)
        # patch output
        layers = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(layers)
        patch_out = Activation('sigmoid')(layers)
        # define model
        self.d_model = Model([in_src_image, in_target_image], patch_out)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        self.d_model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])

    @staticmethod
    def encoder_block(in_layer, filters_num, batch_norm=True):
        init = RandomNormal(stddev=0.02)
        # downsampling layer
        enc = Conv2D(filters_num, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_layer)
        if batch_norm:
            enc = BatchNormalization()(enc, training=True)
        enc = LeakyReLU(alpha=0.2)(enc)
        return enc

    @staticmethod
    def decoder_block(in_layer, skip_conn, filters_num, dropout=True):
        init = RandomNormal(stddev=0.02)
        # add upsampling layer
        dec = Conv2DTranspose(filters_num, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_layer)
        dec = BatchNormalization()(dec, training=True)
        if dropout:
            dec = Dropout(0.5)(dec, training=True)
        # merge with skip connection
        dec = Concatenate()([dec, skip_conn])
        dec = Activation('relu')(dec)
        return dec

    # define the standalone generator model
    def create_generator(self):
        init = RandomNormal(stddev=0.02)
        inp_image = Input(shape=self.image_shape)
        # encoder model
        e1 = self.encoder_block(inp_image, 64, batch_norm=False)
        e2 = self.encoder_block(e1, 128)
        e3 = self.encoder_block(e2, 256)
        e4 = self.encoder_block(e3, 512)
        e5 = self.encoder_block(e4, 512)
        e6 = self.encoder_block(e5, 512)
        e7 = self.encoder_block(e6, 512)
        # bottleneck, no batch norm and relu
        b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
        b = Activation('relu')(b)
        # decoder model
        d1 = self.decoder_block(b, e7, 512)
        d2 = self.decoder_block(d1, e6, 512)
        d3 = self.decoder_block(d2, e5, 512)
        d4 = self.decoder_block(d3, e4, 512, dropout=False)
        d5 = self.decoder_block(d4, e3, 256, dropout=False)
        d6 = self.decoder_block(d5, e2, 128, dropout=False)
        d7 = self.decoder_block(d6, e1, 64, dropout=False)
        # output
        g = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
        out_image = Activation('tanh')(g)
        # define model
        self.g_model = Model(inp_image, out_image)

    # define the combined generator and discriminator model, for updating the generator
    def create_gan(self):
        # make weights in the discriminator not trainable
        for layer in self.d_model.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False
        src_input = Input(shape=self.image_shape)
        # connect the source image to the generator input
        gen_out = self.g_model(src_input)
        # connect the source input and generator output to the discriminator input
        dis_out = self.d_model([src_input, gen_out])
        # src image as input, generated image and classification output
        self.gan_model = Model(src_input, [dis_out, gen_out])
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        self.gan_model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])

    # select a batch of random samples, returns images and target for the discriminator
    @staticmethod
    def generate_real_samples(dataset, n_samples, patch_shape):
        train_poses, train_images = dataset
        # choose random instances
        ix = np.random.randint(0, train_poses.shape[0], n_samples)
        # retrieve selected images
        X1, X2 = train_poses[ix], train_images[ix]
        # generate 'real' class labels (1)
        y = np.ones((n_samples, patch_shape, patch_shape * 2, 1))
        return [X1, X2], y

    # generate a batch of images, returns images and targets
    def generate_fake_samples(self, samples, patch_shape):
        # generate fake instance
        X = self.g_model.predict(samples)
        # create 'fake' class labels (0)
        y = np.zeros((len(samples), patch_shape, patch_shape * 2, 1))
        return X, y

    def load_models(self, d_model_name, g_model_name, gan_model_name):
        self.d_model = load_model(d_model_name)
        self.g_model = load_model(g_model_name)
        self.gan_model = load_model(gan_model_name)

    # generate samples and save as a plot and save the model
    def summarize_performance(self, step, dataset, n_samples=2):
        # select a sample of input images
        [X_realA, X_realB], _ = self.generate_real_samples(dataset, n_samples, 1)
        # generate a batch of fake samples
        fake_labels, _ = self.generate_fake_samples(X_realA, 1)
        # scale all pixels from [-1,1] to [0,1]
        X_realA = (X_realA + 1) / 2.0
        X_realA = X_realA.astype(np.uint8)
        X_realB = (X_realB + 1) / 2.0
        X_realB = X_realB.astype(np.uint8)
        fake_labels = (fake_labels + 1) / 2.0
        fake_labels = fake_labels.astype(np.uint8)
        # plot real pose images
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + i)
            plt.axis('off')
            plt.imshow(X_realA[i])
        # plot generated target image
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + n_samples + i)
            plt.axis('off')
            plt.imshow(fake_labels[i])
        # plot real target image
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + n_samples * 2 + i)
            plt.axis('off')
            plt.imshow(X_realB[i])
        # save plot to file
        path = 'plots/' + str(self.run_number)
        if not os.path.exists(path):
            os.makedirs(path)
        filename1 = path + '/plot_%06d.png' % (step + 1)
        plt.savefig(filename1)
        plt.close()
        # save the generator model
        path = 'models/' + str(self.run_number)
        if not os.path.exists(path):
            os.makedirs(path)
        filename2_g_model = path + '/model2_g_%06d.h5' % (step + 1)
        filename2_d_model = path + '/model2_d_%06d.h5' % (step + 1)
        filename2_gan_model = path + '/model2_gan_%06d.h5' % (step + 1)
        global last_saved_model
        last_saved_model = filename2_g_model
        self.g_model.save(filename2_g_model)
        self.d_model.save(filename2_d_model)
        self.gan_model.save(filename2_gan_model)
        print('>Saved: %s and %s and %s and %s' % (filename1, filename2_g_model, filename2_d_model, filename2_gan_model))

    # train pix2pix model
    def train(self, dataset, n_epochs=100, n_batch=1):
        # determine the output square shape of the discriminator
        n_patch = self.d_model.output_shape[1]
        train_poses, train_images = dataset
        # batch_per_epoch = number of batches per training epoch
        batch_per_epoch = int(len(train_poses) / n_batch)
        # n_steps = number of training iterations
        n_steps = batch_per_epoch * n_epochs
        # manually enumerate epochs
        for i in range(n_steps):
            # select a batch of real samples
            [X_realA, X_realB], y_real = self.generate_real_samples(dataset, n_batch, n_patch)
            # generate a batch of fake samples
            X_fakeB, y_fake = self.generate_fake_samples(X_realA, n_patch)
            # update discriminator for real samples
            d_loss1 = self.d_model.train_on_batch([X_realA, X_realB], y_real)
            # update discriminator for generated samples
            d_loss2 = self.d_model.train_on_batch([X_realA, X_fakeB], y_fake)
            # update the generator
            g_loss, _, _ = self.gan_model.train_on_batch(X_realA, [y_real, X_realB])
            # summarize performance
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))
            # summarize model performance
            if (i + 1) % (batch_per_epoch * 10) == 0:
                self.summarize_performance(i, dataset)


def load_compressed_dataset(filename):
    data = np.load(filename)
    # unpack arrays
    X1 = data['arr_0']
    # scale from [0, 255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    return X1


def save_samples(rn, src_imgs, gen_imgs, tar_imgs):
    path = 'testing_samples/' + str(rn)
    if not os.path.exists(path):
        os.makedirs(path)
    n = len(src_imgs)
    for i in range(n):
        img = (src_imgs[i] + 1) / 2.0 * 255
        img = img.astype(np.uint8)
        cv.imwrite(path + '/' + str(i) + 'src_image.jpg', img)

        img = (gen_imgs[i] + 1) / 2.0 * 255
        img = img.astype(np.uint8)
        cv.imwrite(path + '/' + str(i) + 'gen_image.jpg', img)

        img = (tar_imgs[i] + 1) / 2.0 * 255
        img = img.astype(np.uint8)
        cv.imwrite(path + '/' + str(i) + 'tar_image.jpg', img)


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    data_ready = input("\n\nEnter 'y' if data is ready in a npz file, 'n' if not: ") == 'y'
    subject_name = input('Enter subject name: ')
    run_number = input('Enter run number: ')
    train = input("Enter 'y' to train, 'n' for not: ") == 'y'
    test = input("Enter 'y' to test, 'n' if : ") == 'y'
    first_run = input("Enter 'y' if first run for subject, 'n' if there is pre-saved GAN models: ") == 'y'

    st1 = int(input('Enter beginning index of training data: '))
    en1 = int(input('Enter size of training data: ')) + st1
    st2 = int(input('Enter beginning index of testing data: '))
    en2 = int(input('Enter size of testing data: ')) + st2

    n_ep = int(input("Enter number of epochs: "))

    if data_ready:
        train_filename1 = 'compressed_data/' + subject_name + '_train_images' + str(st1) + '-' + str(en1) + '.npz'
        train_filename2 = 'compressed_data/' + subject_name + '_train_poses' + str(st1) + '-' + str(en1) + '.npz'
        val_filename1 = 'compressed_data/' + subject_name + '_val_images' + str(st2) + '-' + str(en2) + '.npz'
        val_filename2 = 'compressed_data/' + subject_name + '_val_poses' + str(st2) + '-' + str(en2) + '.npz'
    else:
        print('\nReading data...\n')
        train_filename1, train_filename2, val_filename1, val_filename2 = read_and_save(subject_name, st1, en1, st2, en2)

    global last_saved_model
    last_saved_model = None
    if not first_run:
        d_model_name = 'models/' + input("Enter discriminator model name: ")
        last_saved_model = g_model_name = 'models/' + input("Enter generator model name: ")
        gan_model_name = 'models/' + input("Enter GAN model name: ")

    if train:
        X1_images = load_compressed_dataset(train_filename1)
        X2_poses = load_compressed_dataset(train_filename2)
        loaded_data = [X2_poses, X1_images]
        print('Loaded', loaded_data[0].shape, loaded_data[1].shape)
        shape = loaded_data[0].shape[1:]
        print('Image shape is', shape)

        GANModel = Pix2PixModel(run_number, shape)
        print(GANModel.name(), '\n\n')

        if first_run:
            GANModel.create_discriminator()
            GANModel.create_generator()
            GANModel.create_gan()
        else:
            GANModel.load_models(d_model_name, g_model_name, gan_model_name)

        GANModel.train(loaded_data, n_epochs=n_ep)

        # Show sample of output from training dataset
        X1, X2 = loaded_data
        del loaded_data

    if test:
        if last_saved_model is None:
            last_saved_model = 'models/' + input('Enter g_model name: ')
        gen_model = load_model(last_saved_model)

        XX2_images = load_compressed_dataset(val_filename1)
        XX1_poses = load_compressed_dataset(val_filename2)

        gen_model.summary()

        gen_images = gen_model.predict(XX1_poses, batch_size=16)

        save_samples(run_number, XX1_poses, gen_images, XX2_images)

