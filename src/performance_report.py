import os
from markdown2 import markdown

head_string = f"""
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{os.path.basename(os.path.dirname(os.getcwd()))}</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
</head>

<body>
    <div class="container">
        <ul class="nav nav-tabs" id="myTab" role="tablist">
            <li class="nav-item">
                <a class="nav-link active" id="home-tab" data-toggle="tab" href="#home" role="tab" aria-controls="home" aria-selected="true">Home</a>
            </li>
"""

tab_string = """
            <li class="nav-item">
                <a class="nav-link" id="tab-{}" data-toggle="tab" href="#pane-{}" role="tab" aria-controls="pane-{}" aria-selected="false">{}</a>
            </li>
"""

tab_end_string = f"""
        </ul>
        <div class="tab-content" id="myTabContent">
            <div class="tab-pane fade show active" id="home" role="tabpanel" aria-labelledby="home-tab">
                <div class="jumbotron jumbotron-fluid">
                    <div class="container">
                        {markdown(open('../README.md', 'r').read())}
                        <hr class="my-4">
                        <p class="lead">This is a simple web page, for displaying the performances of the various models trained.</p>
                        <p>Navigate using the tabs to check the performance of any model</p>
                    </div>
                </div>
            </div>
"""

tail_string = """
        </div>
    </div>
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
</body>

</html>
"""

def generate_content(i, dirname):
    directory = os.path.join('../performance', dirname)
    contents = os.listdir(directory)
    retstring = f"""
    <div class="tab-pane fade" id="pane-{i+1}" role="tabpanel" aria-labelledby="tab-{i+1}">
    """
    if 'Regression-Report.txt' in contents:
        retstring += f"""
                <h1>Regression Report</h1>
                {open(os.path.join(directory, 'Regression-Report.txt')).read()}
                <div class="row">
                    <div class="col-md-6">
                        <h2>Error Distribution</h2>
                        <img src="{os.path.join(directory, 'Error-Distribution.png')}" width="100%">
                    </div>
                    <div class="col-md-6">
                        <h2>Actual Vs Predicted</h2>
                        <img src="{os.path.join(directory, 'Actual-Vs-Predicted.png')}" width="100%">
                    </div>
                </div>
        """
    elif 'Classification-Report.txt' in contents:
        if 'Preprocessing-Report.txt' in contents:
            retstring += f"""
            <h1>Preprocessing Report</h1>
            {open(os.path.join(directory, 'Preprocessing-Report.txt')).read()}
            <img src="{os.path.join(directory, 'Vocab.png')}" width="100%">
            """
        retstring += f"""
                <h1>Classification Report</h1>
                {open(os.path.join(directory, 'Classification-Report.txt')).read()}
                <h2>Confusion Matrix</h2>
        """
        images = list(filter(lambda x: x.startswith('Confusion-Matrix'), contents))
        if len(images) == 1:
            retstring += f"""
                <img src="{os.path.join(directory, 'Confusion-Matrix.png')}" maxwidth="100%">
        """
        else:
            retstring += """
                <div class="row">
        """
            for cm in images:
                retstring += f"""
                    <div class="col-md-6"><img src="{os.path.join(directory, cm)}" maxwidth="100%"></div>
        """
            retstring += """
                </div>
        """
    elif 'arch.png' in contents:
        retstring += f"""
        <h1>Model Architecture</h1>
        <img src="{os.path.join(directory, 'arch.png')}" maxwidth="100%">
        <h1>Model Performance</h1>
        <img src="{os.path.join(directory, 'Model-Performance.png')}" width="100%">
    """
    return retstring + '\n</div>'

def generate_performance_report():
    with open('../performance/index.html', 'w') as f:
        f.write(head_string)
        model_dirs = [name for name in os.listdir('../performance') if os.path.isdir(os.path.join('../performance', name))]
        for i, d in enumerate(model_dirs):
            f.write(tab_string.format(i+1, i+1, i+1, d))
        f.write(tab_end_string)    
        for i, d in enumerate(model_dirs):
            f.write(generate_content(i, d))
        f.write(tail_string)        
            
if __name__ == '__main__'            :
    generate_performance_report()