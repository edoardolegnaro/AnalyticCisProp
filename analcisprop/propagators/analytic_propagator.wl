ToNUMPY[IN_]:=Module[{expr},
expr = StringReplace[ToString[InputForm[IN]],{"*^"->"e"}];
StringReplace[expr,{"Cos"->"np.cos","ArcSin"->"np.arcsin", "Sin"->"np.sin","Sqrt"->"np.sqrt","["->"(","]"->")","/"->"/","^"->"**","*"->"*"}]
]

(* Constants *)  

GMLval=4.9028001224453001*^3*(24*60*60)^2;  
GMEval=3.986004407799724*^5 *(24*60*60)^2;  
RLval= 1738.;
sostval={GML->GMLval,RL->RLval,GME->GMEval };
Lval=Sqrt[GML*a];

(* Import Analytic Forumlae *)
dir=FileNameJoin[{DirectoryName[$InputFileName],"AnalyticFormulae"}];
heqoldnew=Get[FileNameJoin[{dir,"heqvarold2.wl"}]];
keqoldnew=Get[FileNameJoin[{dir,"keqvarold2.wl"}]];
qeqoldnew=Get[FileNameJoin[{dir,"qeqvarold2.wl"}]];
peqoldnew=Get[FileNameJoin[{dir,"peqvarold2.wl"}]];
sostfreq=Get[FileNameJoin[{dir,"sostfreq2.wl"}]];
sostsolnew = Get[FileNameJoin[{dir,"sostsolnew.wl"}]];
condnewin = Get[FileNameJoin[{dir,"condnewin.wl"}]];

simplifyheqkeqqeqpeqanal[condin0_]:=Module[{heqvaroldnew, keqvaroldnew,qeqvaroldnew, peqvaroldnew},
Lin=Lval/.sostval/.a->ain/.condin0;
asin=((Lin^2/GML)/.sostval)/RLval;
heqvaroldnew=Chop[heqoldnew/.sostfreq/.asn->as/.as->asin];
keqvaroldnew=Chop[keqoldnew/.sostfreq/.asn->as/.as->asin];
qeqvaroldnew=Chop[qeqoldnew/.sostfreq/.asn->as/.as->asin];
peqvaroldnew=Chop[peqoldnew/.sostfreq/.asn->as/.as->asin];
Return[{heqvaroldnew,keqvaroldnew,qeqvaroldnew,peqvaroldnew }]]

(* Compute Analytic Solution *)
computeanalytical[condin0_]:=Module[{ condnewin1, condnewintot, condnewin2, sostsolnewval, condoldin, heqvaranal, keqvaranal, qeqvaranal, peqvaranal, eanal, ianaldeg, newanalytic},
condoldin={as->((Lval^2/GML)/.sostval)/RLval/.a->ain, cosi->Cos[inclin], sini->Sin[inclin], e->ein, incl->inclin, eta->Sqrt[1-ein^2],  g->gin, h->hin, I1->I1in, I2->I2in, I3->I3in, I4->I4in, phi1->phi1in, phi2->phi2in,phi3->phi3in,phi4->phi4in }/.condin0;
condnewin1=Chop[condnewin/.sostfreq/.condoldin];
condnewin2={en0->Sqrt[heqvarn0^2+keqvarn0^2], etan0->Sqrt[1-heqvarn0^2-keqvarn0^2], cihn0->Sqrt[1-peqvarn0^2-qeqvarn0^2], sihn0->Sqrt[peqvarn0^2+qeqvarn0^2], gplushn0->ArcTan[heqvarn0,keqvarn0], hn0->ArcTan[qeqvarn0,peqvarn0]}/.condnewin1;
condnewintot=Join[condnewin1, condnewin2];
sostsolnewval=sostsolnew/.condnewintot;
newanalytic=simplifyheqkeqqeqpeqanal[condin0];
heqvaranal=Chop[newanalytic[[1]]/.sostsolnewval];
keqvaranal=Chop[newanalytic[[2]]/.sostsolnewval];
qeqvaranal=Chop[newanalytic[[3]]/.sostsolnewval];
peqvaranal=Chop[newanalytic[[4]]/.sostsolnewval];
eanal=Sqrt[Chop[Expand[heqvaranal^2+keqvaranal^2]]];
ianaldeg=2*ArcSin[Sqrt[Chop[Expand[qeqvaranal^2+peqvaranal^2]]]]*180./Pi;
Return[{eanal, ianaldeg}]
]

AnalyticSol[a0_, e0_, incl0_, h0deg_, g0deg_, M0deg_] := Module[{condin0,eianal,eanal,ianaldeg},
condin0={ain->a0,ein->e0, inclin->incl0*Degree, Min->M0deg*Degree,  gin->g0deg*Degree,hin->h0deg*Degree,   I1in->0., I2in->0, I3in->0., I4in->0.,phi1in->0.*Degree, phi2in->0.*Degree,phi3in->0.*Degree,phi4in->0.*Degree};
eianal=computeanalytical[condin0];
eanal=ToNUMPY[eianal[[1]]];
ianaldeg=ToNUMPY[eianal[[2]]];
Return[{eanal, ianaldeg}]
]
