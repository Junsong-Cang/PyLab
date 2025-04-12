import matplotlib.pyplot as plt
from PyPDF2 import PdfReader, PdfWriter
from matplotlib.backends.backend_pdf import PdfPages

# Create a plot using Matplotlib
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot Title')
plt.grid(True)

# Save the plot as an image
plt.savefig('plot.pdf')

# Open the existing PDF file
input_pdf = PdfReader(open('/Users/cangtao/Desktop/tmp_tau.pdf', 'rb'))
output_pdf = PdfWriter()

# Add the image to the PDF file
with open('plot.pdf', 'rb') as image_file:
    image_page = PdfReader(image_file)
    output_pdf.add_page(image_page.pages[0])

    # Add the existing PDF content to the output PDF
    for page_num in range(len(image_page.pages)):
        output_pdf.add_page(input_pdf.pages[page_num])

# Save theoutput PDF file with the plot added
with open('/Users/cangtao/Desktop/tmp_tau.pdf', 'wb') as output_file:
    output_pdf.write(output_file)
