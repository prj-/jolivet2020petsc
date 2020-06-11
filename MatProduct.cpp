#include <petsc.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  Mat            A, B, C, D;
  PetscInt       nbs = 10, ntype = 10, nN = 8, m, M, trial = 5;
  PetscViewer    viewer;
  PetscInt       *bs,*N;
  char           **type;
  PetscMPIInt    size;
  PetscBool      flg;
  char           file[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);CHKERRQ(ierr);
  if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only");
  ierr = PetscOptionsGetString(nullptr, nullptr, "-f", file, PETSC_MAX_PATH_LEN, &flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Must indicate binary file with the -f option");
  ierr = PetscOptionsGetInt(nullptr, nullptr, "-trial", &trial, &flg);CHKERRQ(ierr);
  bs = new PetscInt[nbs]();
  ierr = PetscOptionsGetIntArray(nullptr, nullptr, "-bs", bs, &nbs, &flg);CHKERRQ(ierr);
  if (!flg) {
    nbs = 1;
    *bs = 1;
  }
  N = new PetscInt[nN]();
  ierr = PetscOptionsGetIntArray(nullptr, nullptr, "-N", N, &nN, &flg);CHKERRQ(ierr);
  if (!flg) {
    nN = 8;
    N[0] = 1;  N[1] = 2;  N[2] = 4;  N[3] = 8;
    N[4] = 16; N[5] = 32; N[6] = 64; N[7] = 128;
  }
  type = new char*[ntype]();
  ierr = PetscOptionsGetStringArray(nullptr, nullptr, "-type", type, &ntype, &flg);CHKERRQ(ierr);
  if (!flg) {
    ntype = 1;
    *type = new char[6];
    strcpy(*type, std::string(MATSEQAIJ).c_str());
  }
  for (int j = 0; j < nbs; ++j) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &viewer);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatLoad(A, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQAIJ, MATMPIAIJ, "");CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Must indicate a MatAIJ input matrix");
    ierr = MatGetLocalSize(A, &m, nullptr);CHKERRQ(ierr);
    ierr = MatGetSize(A, &M, nullptr);CHKERRQ(ierr);
    ierr = MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);CHKERRQ(ierr);
    if (bs[j] > 1) {
      Mat               T, B;
      const PetscScalar *ptr;
      PetscScalar       *val, *Aa;
      const PetscInt    *Ai, *Aj;
      PetscInt          An;
      PetscBool         done;
      ierr = MatCreateDense(PETSC_COMM_SELF, bs[j], bs[j], bs[j], bs[j], nullptr, &T);CHKERRQ(ierr);
      ierr = MatSetRandom(T, nullptr);CHKERRQ(ierr);
      ierr = MatDenseGetArrayRead(T, &ptr);CHKERRQ(ierr);
      ierr = MatGetRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &An, &Ai, &Aj, &done);CHKERRQ(ierr);
      if (!done || An != m) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Inconsistent sizes");
      ierr = MatSeqAIJGetArray(A, &Aa);CHKERRQ(ierr);
      ierr = MatCreate(PETSC_COMM_WORLD, &B);CHKERRQ(ierr);
      ierr = MatSetType(B, MATSEQBAIJ);CHKERRQ(ierr);
      ierr = MatSetSizes(B, bs[j] * An, bs[j] * An, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
      val = new PetscScalar[Ai[An] * bs[j] * bs[j]];
      for(int i = 0; i < Ai[An]; ++i)
        for(int k = 0; k < bs[j] * bs[j]; ++k)
          val[i * bs[j] * bs[j]] = Aa[i] * ptr[k];
      ierr = MatSeqBAIJSetPreallocationCSR(B, bs[j], Ai, Aj, val);CHKERRQ(ierr);
      delete [] val;
      ierr = MatSeqAIJRestoreArray(A, &Aa);CHKERRQ(ierr);
      ierr = MatRestoreRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &An, &Ai, &Aj, &done);CHKERRQ(ierr);
      ierr = MatDenseRestoreArrayRead(T, &ptr);CHKERRQ(ierr);
      ierr = MatDestroy(&T);CHKERRQ(ierr);
      ierr = MatDestroy(&A);CHKERRQ(ierr);
      A = B;
      ierr = MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);CHKERRQ(ierr);
    }
    for (int i = 0; i < ntype; ++i) {
      std::vector<std::string> list ({ "aij", "baij", "sbaij", "seqaij", "seqbaij", "seqsbaij" });
      // TODO FIXME aijcusparse, seqaijcusparse
      std::vector<std::string>::const_iterator it = std::find(list.cbegin(), list.cend(), type[i]);
      if (it != list.cend()) {
        ierr = MatConvert(A, type[i], MAT_INPLACE_MATRIX, &A);CHKERRQ(ierr);
      } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_UNKNOWN_TYPE, "Not implemented");
      for(int k = 0; k < nN; ++k) {
        ierr = MatCreateDense(PETSC_COMM_WORLD, bs[j] * m, PETSC_DECIDE, bs[j] * M, N[k], nullptr, &C);CHKERRQ(ierr);
        ierr = MatCreateDense(PETSC_COMM_WORLD, bs[j] * m, PETSC_DECIDE, bs[j] * M, N[k], nullptr, &D);CHKERRQ(ierr);
        ierr = MatSetRandom(C, nullptr);CHKERRQ(ierr);
        // TODO FIXME MatConvert(SeqDenseCuda), don't know if there is an efficient random number generator on GPU in PETSc right now?
        PetscLogStage stage;
        ierr = PetscLogStageRegister(std::string("type_" + std::string(type[i]) + "-bs_" + std::to_string(bs[j]) + "-N_" + std::to_string(N[k])).c_str(), &stage);CHKERRQ(ierr);
        if (N[k] > 1) {
          ierr = MatProductCreateWithMat(A, C, NULL, D);CHKERRQ(ierr);
          ierr = MatProductSetType(D, MATPRODUCT_AB);CHKERRQ(ierr);
          ierr = MatProductSetFromOptions(D);CHKERRQ(ierr);
          ierr = MatProductSymbolic(D);CHKERRQ(ierr);
          ierr = MatProductNumeric(D); CHKERRQ(ierr);
          ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
          for (int m = 0; m < trial; ++m) {
            ierr = MatProductNumeric(D); CHKERRQ(ierr);
          }
          ierr = PetscLogStagePop();CHKERRQ(ierr);
        } else {
          Vec cC, cD;
          ierr = MatDenseGetColumnVecRead(C, 0, &cC);CHKERRQ(ierr);
          ierr = MatDenseGetColumnVecWrite(D, 0, &cD);CHKERRQ(ierr);
          ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
          for (int m = 0; m < trial; ++m) {
            ierr = MatMult(A, cC, cD); CHKERRQ(ierr);
          }
          ierr = PetscLogStagePop();CHKERRQ(ierr);
          ierr = MatDenseRestoreColumnVecRead(C, 0, &cC);CHKERRQ(ierr);
          ierr = MatDenseRestoreColumnVecWrite(D, 0, &cD);CHKERRQ(ierr);
        }
        ierr = MatDestroy(&C);CHKERRQ(ierr);
        ierr = MatDestroy(&D);CHKERRQ(ierr);
      }
    }
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }
  for (int i = 0; i < ntype; ++i) {
    ierr = PetscFree(type[i]);CHKERRQ(ierr);
  }
  delete [] type;
  delete [] N;
  delete [] bs;
  ierr = PetscFinalize();
  return 0;
}
