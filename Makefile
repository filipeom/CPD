CC=mpicc
SRCDIR=src
BUILDDIR=build
TARGET=bin/matFact

SRCEXT=c
SOURCES=$(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS=$(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CFLAGS=-g -Wall -std=gnu11 -O0 -fopenmp
LINKER=-fopenmp -lm


$(TARGET): $(OBJECTS)
	@echo "Linking...";
	@echo "$(CC) $^ -o $(TARGET) $(LINKER)"; $(CC) $^ -o $(TARGET) $(LINKER)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	@echo "$(CC) $(CFLAGS) -c -o $@ $<"; $(CC) $(CFLAGS) -c -o $@ $<

clean:
	@echo "Cleaning...";
	@echo "$(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET)

.PHONY: clean
