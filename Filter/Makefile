#--------------------------------------------------------------------
MEXT=mexa64
MATLABPATH=/matlab/r2006a
MATLABLIBDIR=glnxa64
#--------------------------------------------------------------------
COMPILEFLAGS=-fPIC -fno-omit-frame-pointer -ansi -D_GNU_SOURCE
OPTIMFLAGS=-O3 -DNDEBUG
DEBUGFLAGS=-g -DDEBUG
#--------------------------------------------------------------------
LINKFLAGS=-pthread -shared -Wl,--version-script,$(MATLABPATH)/extern/lib/$(MATLABLIBDIR)/mexFunction.map -Wl,-rpath-link,$(MATLABPATH)/bin/$(MATLABLIBDIR)
LIBDIRS=-L$(MATLABPATH)/bin/$(MATLABLIBDIR)
LIBS=-lmx -lmex -lmat -lm
INCLUDE=-I$(MATLABPATH)/extern/include
#--------------------------------------------------------------------
all:
	clear
	icpc $(COMPILEFLAGS) $(LINKFLAGS) $(OPTIMFLAGS) $(LIBDIRS) $(LIBS) $(INCLUDE) -DMATLAB_MEX_FILE -o mdwt.$(MEXT) mdwt.cpp mdwt_r.cpp
	icpc $(COMPILEFLAGS) $(LINKFLAGS) $(OPTIMFLAGS) $(LIBDIRS) $(LIBS) $(INCLUDE) -DMATLAB_MEX_FILE -o midwt.$(MEXT) midwt.cpp midwt_r.cpp
	icpc $(COMPILEFLAGS) $(LINKFLAGS) $(OPTIMFLAGS) $(LIBDIRS) $(LIBS) $(INCLUDE) -DMATLAB_MEX_FILE -o mrdwt.$(MEXT) mrdwt.cpp mrdwt_r.cpp
	icpc $(COMPILEFLAGS) $(LINKFLAGS) $(OPTIMFLAGS) $(LIBDIRS) $(LIBS) $(INCLUDE) -DMATLAB_MEX_FILE -o mirdwt.$(MEXT) mirdwt.cpp mirdwt_r.cpp

clean:
