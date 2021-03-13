  def get_unit_vectors(number, dimension): # Distribution is not uniform
    r = (np.random.rand(number, dimension) - 0.5)*2
    long_ones = np.linalg.norm(r, axis = 1) > 1
    while np.sum(long_ones):
      r[long_ones] = np.random.rand(np.sum(long_ones), dimension)
      long_ones = np.linalg.norm(r, axis = 1) > 1

    for v in r:
      v /= np.linalg.norm(v)
    return r


def get_line_data(num_of_lines = 2, dimensions = 3, segment_length_limits = (2, 10), displacement_length_limits = (0, 3), min_max_points = (50, 100), noise_magnitude = 0.1, show = True):
  directions = get_unit_vectors(num_of_lines, dimensions)
  segment_lengths = np.random.rand(num_of_lines)*(segment_length_limits[1] - segment_length_limits[0]) + segment_length_limits[0]
  displacement_lengths = np.random.rand(num_of_lines, 1)*(displacement_length_limits[1] - displacement_length_limits[0]) + displacement_length_limits[0]
  displacement_vectors = get_unit_vectors(num_of_lines, dimensions)*displacement_lengths
  num_of_points = np.random.randint(low = min_max_points[0], high = min_max_points[1], size = (num_of_lines))


  random_points = []
  for i in range(num_of_lines):
    random_points.append(np.outer((np.random.rand(num_of_points[i]) - 0.5)*segment_lengths[i], directions[i]) + displacement_vectors[i] + (np.random.rand(num_of_points[i], dimensions)-0.5)*noise_magnitude)

  if show and dimensions == 3:
    fig = plt.figure()
    ax = Axes3D(fig)
    for group in random_points:
      color = np.random.random(3)
      ax.scatter(group[:, 0], group[:, 1], group[:, 2], color=color)
  elif show and dimensions == 2:
    fig = plt.figure()
    for group in random_points:
      color = np.random.random(3)
      plt.scatter(group[:, 0], group[:, 1], color=color)

  xs = None
  classes = []
  for class_, (num, group) in enumerate(zip(num_of_points, random_points)):
    xs = np.append(xs, group, axis = 0) if type(xs) == type(group) else np.copy(group)
    classes += [class_]*num

  classes = np.array(classes)
  shape = (classes.size, num_of_lines)
  ys = np.zeros(shape)
  rows = np.arange(classes.size)
  ys[rows, classes] = 1

  return (xs, ys)
  
def get_sphere_data(num_of_spheres = 2, dimensions = 3, min_max_radii = (0, 6), min_max_points = (200, 300), noise_magnitude = 0.1, show = True):  
  radii = np.random.rand(num_of_spheres)*(min_max_radii[1] - min_max_radii[0]) + min_max_radii[0]
  num_of_points = np.random.randint(low = min_max_points[0], high = min_max_points[1], size = (num_of_spheres))

  random_points = []
  for i in range(num_of_spheres):
    random_points.append(get_unit_vectors(num_of_points[i], dimensions)*radii[i] + (np.random.rand(num_of_points[i], dimensions)-0.5)*noise_magnitude)

  if show and dimensions == 3:
    fig = plt.figure()
    ax = Axes3D(fig)
    for group in random_points:
      color = np.random.random(3)
      ax.scatter(group[:, 0], group[:, 1], group[:, 2], color=color)
  elif show and dimensions == 2:
    fig = plt.figure()
    for group in random_points:
      color = np.random.random(3)
      plt.scatter(group[:, 0], group[:, 1], color=color)

  xs = None
  classes = []
  for class_, (num, group) in enumerate(zip(num_of_points, random_points)):
    xs = np.append(xs, group, axis = 0) if type(xs) == type(group) else np.copy(group)
    classes += [class_]*num

  classes = np.array(classes)
  shape = (classes.size, num_of_spheres)
  ys = np.zeros(shape)
  rows = np.arange(classes.size)
  ys[rows, classes] = 1

  return (xs, ys)


def get_model(inp_dim, num_of_classes, num_of_cells, hidden_activation = "relu", output_activation = "softmax", summary = True):
  """
  This function gets the input dimensions, number of classes, number of cells and hidden activation function and returns the fully connected model.
  The default hidden activation function is relu and the default output activation function is softmax.
  """
  num_of_layers = len(num_of_cells) + 1
  model = Sequential()
  model.add(Dense(num_of_cells[0], input_dim = inp_dim, activation=hidden_activation))

  for i in range(1, num_of_layers - 1):
    model.add(Dense(num_of_cells[i], activation=hidden_activation))

  model.add(Dense(num_of_classes, activation=output_activation))

  if summary:
    model.summary()
  
  return model
  
def get_weight_history(XTraining, XValidation, YTraining, YValidation, model, batch_size, epochs, in_a_row, loss='binary_crossentropy', metrics=['accuracy'], limit = 1):
  """
  This function takes the training and test data, model parameters and training metrics.
  The training stops when validation accuracy is above the given limit or the epoch limit is reached.
  The default loss function is binary_crossentropy.
  """
  def store_weights(weights, biases, loss):
    w_b = model.get_weights()
    weights.append(w_b[::2])
    biases.append(w_b[1::2])
    losses.append(loss)

  model = clone_model(model)
  opt = Adam(learning_rate=0.01)
  model.compile(loss=loss, optimizer=opt, metrics=metrics)

  weights = []
  biases = []
  losses = []

  weight_callback = LambdaCallback(on_batch_end = lambda epoch, logs: store_weights(weights, biases, logs["loss"]))

  count = 0
  for i in range(epochs):
    print(f"Epoch: {i}")
    history = model.fit(XTraining, YTraining, batch_size=batch_size, epochs=1, callbacks = [weight_callback], shuffle = False, validation_data=(XValidation, YValidation))
    if history.history["val_accuracy"][-1] >= limit:
      count += 1
    else:
      count = 0
    if count == in_a_row:
      print(f"Terminated in {i} epochs.")
      return weights, biases, losses

  return [], [], [] 


def vectorize_parameters(weights, biases, losses):
  """
  This function takes weights, biases, and losses and returns a vector.
  """
  columns = []
  for weight, bias, loss in zip(weights, biases, losses):
    column = np.array([])
    for w in weight:
      column = np.append(column, w.flatten())
    for b in bias:
      column = np.append(column, b.flatten())
    column = np.append(column, [loss])
    columns.append(column)
  return np.array(columns).T

def get_matrix_multiple_networks(num_of_networks, num_of_points, XTraining, XValidation, YTraining, YValidation, inp_dim, num_of_classes, num_of_cells, batch_size = 50, epochs = 100, in_a_row = 20, limit = 1):
  """
  This function takes number of different networks, number of points to be saved from the training history, training and test data and training and model parameters;
  And outputs the training paths of each randomly initialized network as a matrix.
  """
  matrix = None
  for i in range(num_of_networks):
    model = get_model(inp_dim = dimensions, num_of_classes = num_of_classes, num_of_cells = num_of_cells)
    weights, biases, losses = get_weight_history(XTraining, XValidation, YTraining, YValidation, model, batch_size=batch_size, epochs=epochs, in_a_row=in_a_row, limit = limit)
    weights, biases, losses = weights[-num_of_points:], biases[-num_of_points:], losses[-num_of_points:]
    vec = vectorize_parameters(weights, biases, losses)
    if isinstance(matrix, type(None)):
      matrix = np.empty(shape = (vec.shape[0], vec.shape[1]*num_of_networks))
    matrix[0:vec.shape[0], vec.shape[1]*i:vec.shape[1]*(i+1)] = vec
  return matrix

