
import torch


def blur(output: torch.Tensor):

  """
  Assumes input of shape n_images x height x width with no one-hot encoding
  Sets every pixel's class to the most numerous class in a 3x3 area around the pixel
  """

  output = output.cpu()

  n_images, height, width = output.shape

  # Performs padding
  padded = torch.zeros(n_images, height+2, width+2)
  padded = padded.to(output.dtype)
  padded[:, 1:-1, 1:-1] = output

  blurred = torch.empty_like(output)
  for i in range(height):
    for j in range(width):
      area = padded[:, i:i+3, j:j+3].reshape(-1, 9)
      for k in range(area.shape[0]):
        most_common = torch.argmax(torch.unique(area[k], return_counts=True)[1])
        blurred[k, i, j] = most_common
  
  return blurred





