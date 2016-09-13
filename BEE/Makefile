CC=gcc
CFLAGS=-fPIC -O3 -lpthread -std=gnu99 -Wall -ffast-math -mtune=native -I./src/mtrand

# http://stackoverflow.com/a/12099167
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
		CFLAGS += -lrt -lm
		LDFLAGS = -Wl,-soname,libBEE.so

		# If it is necessary to create shared libraries from mtrand:
		# CFLAGS += -L./bin/lib -lrt -lm -lrandomkit -ldistributions
		# LFLAGS = -Wl,-soname,libBEE.so,-rpath=./bin/lib
		#	gcc -shared -o ./bin/lib/librandomkit.so ./bin/obj/randomkit.o
		#	gcc -shared -o ./bin/lib/libdistributions.so ./bin/obj/distributions.o ./bin/obj/randomkit.o

endif

ifeq ($(UNAME_S),Darwin)
		LDFLAGS = -Wl,-install_name,libBEE.so
endif


# debug:
# 	CFLAGS += -g

all: compile

compile:
	@echo "Creating directories..."
	mkdir -p ./bin
	mkdir -p ./bin/obj
	mkdir -p ./bin/lib
	@echo "Creating object files..."
	gcc -c -Wall -Werror -fPIC -c ./src/mtrand/distributions.c -o ./bin/obj/distributions.o
	gcc -c -Wall -Werror -fPIC -c ./src/mtrand/randomkit.c -o ./bin/obj/randomkit.o
	@echo "Finalising..."
	$(CC) -shared $(LDFLAGS) -o ./bin/lib/libBEE.so ./src/BEE.c $(CFLAGS) ./bin/obj/randomkit.o ./bin/obj/distributions.o
	@echo "Done!"

install:
	@echo "Installing the C library..."
	cp ./bin/lib/libBEE.so /usr/local/lib/libBEE.so
	@echo "Installing the Python module..."
	python setup.py install
	@echo "Done!"
	@echo "From Python, import BEE to use the simulator."

clean:
	rm -r -f bin
	rm -r -f build
