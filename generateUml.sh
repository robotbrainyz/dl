pyreverse src/
mv *.dot uml/
dot -Tpng uml/classes.dot -o uml/dlt_class_diagram.png
dot -Tpng uml/packages.dot -o uml/dlt_package_diagram.png


