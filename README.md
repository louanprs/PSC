# PSC

Les calculs sont tirés de :
"Design of a supersonic wind tunnel"
https://drive.google.com/drive/folders/1b4oRFMFnW-b5iiJwy8KRgf2TBNfjyPlJ

planar_moc.py
Code le plus abouti et que j'ai le plus commenté.
Applique la méthode des caractéristiques à une gradual expansion nozzle quelconque.
Les caractéristiques sont initialisées par des conditions sur la ligne sonique.

min_len_nozzle.py
Construit la minimum length nozzle à partir du Mach de sortie désiré.
La tuyère présente un angle au col qui crée une détente de Prandtl-Meyer (cf wikipedia)
La méthode employée est particulière : on part des caractéristiques divergeant de l'angle aigu.

