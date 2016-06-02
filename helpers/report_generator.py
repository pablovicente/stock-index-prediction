from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image,Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors

import time
import datetime as datetime



def pdf_tables(algorithms_list, features_list, name):

    filename = name
    
    ## Generate PDF  ##
    styles=getSampleStyleSheet()

    table_data = []
    table_data.append(algorithms_list)#table_data.append(["Nombre", "Influencia", "Sentimiento", "Pais"])


    for index in range(len(features_list)):
        row_html = []
        for cell in features_list[index]:
            cell_html = Paragraph(cell, styles["Normal"])
            row_html.append(cell_html)
        
        table_data.append(row_html)

        #href = 'https://twitter.com/intent/user?user_id=' + str(user['user_id'])
        #link = '<a href=' + href + '>' + user['name'] + '</a>'
        #try:
        #    sentiment = round(user['mean_sentiment'], 2)
        #except:
        #    sentiment = "-"
        #table_data.append([para, round(user['influence'], 2), sentiment, user['country']])


     
    doc = SimpleDocTemplate(filename,pagesize=A4,
                            rightMargin=72,leftMargin=72,
                            topMargin=72,bottomMargin=30)

    # container for the 'Flowable' objects
    elements = []

    logo = "images/logo.jpg"
    formatted_time = time.ctime()
   

    elements.append(Spacer(1, 120))
     
    im = Image(logo, 4*inch, 2*inch)
    elements.append(im)     
    elements.append(Spacer(1, 40))

   
    styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER))       
     
    ptext = '<font size=40>Esperanto</font>'
    elements.append(Paragraph(ptext, styles["Center"]))
    elements.append(Spacer(1, 80))

    ptext = '<font size=20>Informe sobre los usuarios mas influyentes</font>'

    elements.append(Paragraph(ptext, styles["Center"]))
    elements.append(Spacer(1, 40))
    ptext = '<font size=16>' + 'HOLA' + '</font>'
    elements.append(Paragraph(ptext, styles["Center"]))
    elements.append(PageBreak())

    t=Table(table_data, colWidths=None, rowHeights=None)

    t.setStyle(TableStyle(
        [('LINEBELOW', (0,0), (-1,0), 2, colors.black),
        ('ALIGN',(0,0),(-1,-1),'CENTER')]
        ))

    elements.append(t)
    # write the document to disk
    doc.build(elements)